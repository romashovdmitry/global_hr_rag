"""LangGraph node: generate the answer using intent-specific guidance."""

from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node


# Appended to every intent-specific prompt to enforce strict, focused answers.
_CONSTRAINTS = (
    "\n\nStrict output rules:"
    "\n- Answer ONLY what the user asked. Nothing more."
    "\n- Do NOT write interview questions or sample answers."
    "\n- Do NOT write cover letter content or resume advice."
    "\n- Do NOT write in first person as if you are the candidate."
    "\n- Do NOT add disclaimers about data being fictional or for demo purposes."
    "\n- Do NOT add summaries of your own answer at the end."
    "\n- Do NOT add markdown separators (---)."
    "\n- Do NOT add unsolicited suggestions, tips, or next steps."
    "\n- Be concise. If the user asked for one vacancy, give one. If they asked for a list, give a list."
)

_SYSTEM_BY_INTENT = {
    "find_matching": (
        "You are an expert career advisor for IT professionals. "
        "Using ONLY the provided vacancy data, answer the candidate's question. "
        "Present relevant vacancies with role, company, salary, and a brief reason why it fits. "
        "Do NOT invent details. If no vacancies match, say so honestly."
        + _CONSTRAINTS
    ),

    "role_transition": (
        "You are an expert career advisor helping a candidate change careers. "
        "Using ONLY the provided vacancy data, present vacancies the candidate could transition INTO. "
        "For each vacancy briefly explain which transferable skills make them a fit. "
        "Do NOT suggest vacancies in the same specialisation as their current role. "
        "Do NOT invent details."
        + _CONSTRAINTS
    ),

    "compare": (
        "You are an expert career advisor for IT professionals. "
        "The candidate wants to pick the best option. "
        "Using ONLY the provided vacancy data, recommend ONE vacancy and explain concisely why it is the best fit. "
        "Do NOT invent details."
        + _CONSTRAINTS
    ),

    "clarify_vacancy": (
        "You are an expert career advisor for IT professionals. "
        "Answer the candidate's specific question using ONLY the provided vacancy data. "
        "Be direct and concise. Do NOT invent details."
        + _CONSTRAINTS
    ),

    "followup": (
        "You are an expert career advisor for IT professionals. "
        "Answer based on the updated criteria using ONLY the provided vacancy data. "
        "Do NOT repeat information already given in previous messages. "
        "Do NOT invent details."
        + _CONSTRAINTS
    ),

    "aggregate": (
        "You are an expert career advisor performing a data analysis across multiple IT vacancies. "
        "Synthesise the provided vacancy data to answer the user's aggregation or analysis question. "
        "Present clear patterns, ranges, or comparisons. If counts or averages are requested, "
        "compute them from the provided data only. "
        "If the data is insufficient to answer precisely, state what you CAN conclude and note the limitation. "
        "Do NOT invent data or extrapolate beyond what is provided."
        + _CONSTRAINTS
    ),

    "find_matching_ambiguous": (
        "You are an expert career advisor for IT professionals. "
        "The user's request may not have an exact match in the provided vacancy data. "
        "Suggest the closest available options and explicitly note if the exact seniority level, "
        "tech stack, or other requirement is not available. "
        "Do NOT invent details. Do NOT pretend an exact match exists if it doesn't."
        + _CONSTRAINTS
    ),
}


@timed_node("generator")
async def generator_node(state: RAGState) -> dict:
    """Generate the final answer using an intent-specific system prompt.

    Selecting the prompt by intent avoids generic rules that mix concerns and
    conflict with each other; each intent gets a focused, unambiguous instruction.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``generated_answer`` and a status entry.
    """
    vacancies = state.get("graded_vacancies") or []
    intent = state.get("intent", "find_matching")

    if not vacancies:
        unsupported = state.get("unsupported_criteria") or []
        if unsupported:
            criteria_list = ", ".join(unsupported)
            answer = (
                f"Our vacancy database does not contain the following information "
                f"and cannot filter by it: {criteria_list}. "
                f"No other matching vacancies were found for your request either. "
                f"Try asking without those criteria — for example, search by role, "
                f"tech stack, employment type (Remote / Full-time), or salary range."
            )
        else:
            answer = (
                "No suitable vacancies were found for your request. "
                "Try adjusting your criteria or ask in a different way."
            )
        return {
            "generated_answer": answer,
            "status_stream": state.get("status_stream", []) + ["Generating response…"],
        }

    vacancy_block = "\n\n".join(
        f"=== Vacancy {i + 1} ===\n"
        f"Role: {v.get('role', '')}\n"
        f"Company: {v.get('company', '')}\n"
        f"Salary: {v.get('salary') or 'Not specified'}\n"
        f"Status: {', '.join(v.get('status_of_role') or [])}\n"
        f"Contacts: {v.get('contacts', '')}\n"
        f"Description: {v.get('description_of_vacancy', '')}"
        for i, v in enumerate(vacancies)
    )

    history_block = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("chat_history", [])
    )

    user_parts: list[str] = []
    if state.get("cv_text"):
        user_parts.append(f"Candidate CV:\n{state['cv_text']}")
    if intent == "role_transition" and state.get("current_role"):
        user_parts.append(f"Current role (transitioning FROM): {state['current_role']}")

    unsupported = state.get("unsupported_criteria") or []
    if unsupported:
        criteria_list = ", ".join(unsupported)
        user_parts.append(
            f"IMPORTANT — UNSUPPORTED FILTER NOTICE: The user's question contains "
            f"criteria that do NOT exist in our vacancy database: [{criteria_list}]. "
            f"Start your response by explicitly informing the user that our vacancy data "
            f"does not include this information and we cannot filter by it. "
            f"Then present the best matching vacancies based on the remaining criteria."
        )

    if state.get("grounding_feedback"):
        retry_note = (
            "IMPORTANT — PREVIOUS DRAFT WAS REJECTED BY GROUNDEDNESS CHECK:\n"
            f"Reason: {state['grounding_feedback']}\n"
            "Regenerate the answer using only facts that are explicitly present "
            "in the vacancy data below. Remove or soften any unsupported claim."
        )
        if state.get("rejected_answer"):
            retry_note += f"\nRejected draft:\n{state['rejected_answer']}"
        user_parts.append(retry_note)

    user_parts.append(f"Candidate question: {state['user_query']}")
    user_parts.append(f"Vacancy data:\n{vacancy_block}")

    system_prompt = _SYSTEM_BY_INTENT.get(intent) or _SYSTEM_BY_INTENT["find_matching"]
    llm = get_llm_service().get_llm()

    messages = [{"role": "system", "content": system_prompt}]
    if history_block:
        messages.append({"role": "user", "content": f"Previous conversation:\n{history_block}"})
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    response = await llm.ainvoke(messages)

    # Collect token usage — format differs by provider:
    # Groq / OpenAI: response.usage_metadata {"input_tokens", "output_tokens"}
    # Ollama:        response.response_metadata {"prompt_eval_count", "eval_count"}
    usage = response.usage_metadata or {}
    input_tokens: int | None = usage.get("input_tokens")
    output_tokens: int | None = usage.get("output_tokens")
    if input_tokens is None:
        meta = response.response_metadata or {}
        input_tokens = meta.get("prompt_eval_count")
        output_tokens = meta.get("eval_count")

    return {
        "generated_answer": response.content,
        "grounding_feedback": None,
        "rejected_answer": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status_stream": state.get("status_stream", []) + ["Generating response…"],
    }
