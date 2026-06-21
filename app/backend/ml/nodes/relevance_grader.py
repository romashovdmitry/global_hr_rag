"""LangGraph node: batch-grade retrieved vacancies for relevance to the user intent."""

from pydantic import BaseModel
from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node


class RelevanceResult(BaseModel):
    """Structured output: which vacancy numbers pass the relevance threshold.

    Uses simple 1-based integers instead of UUIDs so small models can reliably
    reproduce them without copy-paste errors on long hexadecimal strings.
    """

    relevant_numbers: list[int]


_SYSTEM_BY_INTENT = {
    "find_matching": """You are a senior technical recruiter.
Review the numbered vacancy summaries and return the numbers of those genuinely relevant
to the candidate's query and technical profile.
Include vacancies whose tech stack, role type, or requirements meaningfully overlap.
Be strict: exclude vacancies that are clearly off-topic.
Respond with JSON: {{"relevant_numbers": [1, 3, 5]}}""",

    "role_transition": """You are a senior technical recruiter helping a candidate change careers.
The candidate wants to LEAVE their current role and move into something DIFFERENT.
Current role to EXCLUDE: {current_role}

Rules:
- EXCLUDE any vacancy with the same or very similar role title as the current role.
- INCLUDE vacancies in a different specialisation that could leverage transferable skills.
- If unsure whether a vacancy is in the same category, exclude it.

Respond with JSON: {{"relevant_numbers": [2, 4]}}""",

    "compare": """You are a senior technical recruiter.
The user wants to compare or choose between specific vacancies.
Include vacancies that are closely related to the comparison topic.
Respond with JSON: {{"relevant_numbers": [1, 2]}}""",

    "clarify_vacancy": """You are a senior technical recruiter.
The user is asking about a specific vacancy mentioned earlier.
Include the vacancy that best matches the role/company from the conversation.
Respond with JSON: {{"relevant_numbers": [3]}}""",

    "followup": """You are a senior technical recruiter.
The user is refining their previous request. Apply the updated filters and include
only vacancies that meet the refined requirements.
Respond with JSON: {{"relevant_numbers": [1, 4, 6]}}""",

    "aggregate": """You are a senior technical recruiter performing a broad database analysis.
The user wants to synthesise information across many vacancies.
Include ALL vacancies relevant to the topic — the generator needs a large set.
Only exclude vacancies completely off-topic (different domain entirely).
Respond with JSON: {{"relevant_numbers": [1, 2, 3, 4, 5, 6, 7]}}""",
}


@timed_node("relevance_grader")
async def relevance_grader_node(state: RAGState) -> dict:
    """Filter retrieved vacancies to those relevant for the classified intent.

    Uses an intent-specific grading prompt so that the filtering criteria
    match what the user actually wants (e.g. excluding the current role for
    role_transition, focusing on a specific vacancy for clarify_vacancy).

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``graded_vacancies`` and a status entry.
    """
    vacancies = state["retrieved_vacancies"]
    if not vacancies:
        return {
            "graded_vacancies": [],
            "status_stream": state.get("status_stream", []) + ["No vacancies to grade."],
        }

    intent = state.get("intent", "find_matching")
    system_template = _SYSTEM_BY_INTENT.get(intent, _SYSTEM_BY_INTENT["find_matching"])
    system_prompt = system_template.format(
        current_role=state.get("current_role") or "unknown"
    )

    # Use 1-based integer labels instead of raw UUIDs — small models reliably
    # reproduce short numbers but often mangle long hexadecimal strings.
    index_to_vacancy = {i + 1: v for i, v in enumerate(vacancies)}

    summaries = "\n\n".join(
        f'#{i + 1} | Role: {v.get("role", "")}\n'
        f'Stack: {v.get("description_of_vacancy", "")[:200]}'
        for i, v in enumerate(vacancies)
    )

    cv_snippet = (state.get("cv_text") or "Not provided")[:400]
    context = (
        f"User query: {state['user_query']}\n"
        f"CV: {cv_snippet}\n\n"
        f"Vacancies (use the # number in relevant_numbers):\n{summaries}"
    )

    llm = get_llm_service().get_llm().with_structured_output(RelevanceResult)

    result: RelevanceResult = await llm.ainvoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
    )

    valid_numbers = {n for n in result.relevant_numbers if n in index_to_vacancy}
    graded = [index_to_vacancy[n] for n in sorted(valid_numbers)]

    # Post-filter: exclude vacancies that mention any explicitly excluded skill.
    exclude_skills = [s.lower() for s in (state.get("exclude_skills") or [])]
    if exclude_skills:
        before_count = len(graded)
        graded = [
            v for v in graded
            if not any(
                skill in (v.get("description_of_vacancy") or "").lower()
                or skill in (v.get("role") or "").lower()
                for skill in exclude_skills
            )
        ]
        excluded_count = before_count - len(graded)
        if excluded_count:
            status_suffix = f" (removed {excluded_count} with excluded skills: {', '.join(state['exclude_skills'])})"
        else:
            status_suffix = ""
    else:
        status_suffix = ""

    return {
        "graded_vacancies": graded,
        "status_stream": state.get("status_stream", []) + [
            f"Relevance check: {len(graded)} / {len(vacancies)} vacancies passed…{status_suffix}"
        ],
    }
