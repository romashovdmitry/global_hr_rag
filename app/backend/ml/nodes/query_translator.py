"""LangGraph node: translate the user query into diversified search queries."""

from pydantic import BaseModel
from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node


class SearchQueries(BaseModel):
    """Three diversified search queries targeting different aspects of the user's intent."""

    queries: list[str]


_SYSTEM_BY_INTENT = {
    "find_matching": """Generate 3 distinct search queries to find job vacancies that match the user's
requirements, skills, or CV profile. Each query should target a different angle
(e.g. core tech stack, seniority level, domain) to maximise recall diversity.
Return JSON with a "queries" list of exactly 3 strings.""",

    "role_transition": """The user wants to switch FROM their current role INTO a different one.
Generate 3 distinct search queries that target roles the user could TRANSITION INTO
based on their transferable skills. Do NOT generate queries about their current role.
Focus on adjacent or target roles that make use of their existing skills.
Example: a QA engineer transitioning could move into: backend dev, DevOps, SDET, product.
Return JSON with a "queries" list of exactly 3 strings.""",

    "compare": """The user wants to compare vacancies or choose the best option from ones
already discussed. Generate 3 queries to retrieve those specific vacancies.
Use role names, company names, or skills mentioned in the conversation history.
Return JSON with a "queries" list of exactly 3 strings.""",

    "clarify_vacancy": """The user is asking a follow-up question about a specific vacancy
mentioned earlier in the conversation. Generate 3 queries to retrieve that vacancy
and closely related ones for context. Use the company name, role name, or skills
from the conversation history.
Return JSON with a "queries" list of exactly 3 strings.""",

    "followup": """The user is refining or continuing their previous request. Generate 3
queries that address the updated or narrowed requirements. Use conversation history
to understand what was already discussed and what the refinement adds.
Return JSON with a "queries" list of exactly 3 strings.""",

    "aggregate": """The user wants to aggregate, synthesise, or compare information ACROSS
MULTIPLE vacancies in the database (statistics, patterns, cross-vacancy analysis).
Generate 3 broad search queries that together maximise coverage of the relevant
vacancy space — cast a wide net so the retrieval step can surface many documents.
Avoid queries that are too specific to a single vacancy.
Return JSON with a "queries" list of exactly 3 strings.""",
}


def _format_history(state: RAGState) -> str:
    history = state.get("chat_history") or []
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


@timed_node("query_translator")
async def query_translator_node(state: RAGState) -> dict:
    """Translate the user query into 3 diversified Qdrant search strings.

    The search strategy is selected based on the classified intent so that
    different query types (matching, transition, compare, etc.) produce
    semantically appropriate retrieval queries.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``search_queries`` and a status entry.
    """
    intent = state.get("intent", "find_matching")
    system_prompt = _SYSTEM_BY_INTENT.get(intent, _SYSTEM_BY_INTENT["find_matching"])

    llm = get_llm_service().get_light_llm().with_structured_output(SearchQueries)

    parts: list[str] = []
    if state.get("cv_text"):
        parts.append(f"Candidate CV (first 800 chars):\n{state['cv_text'][:800]}")
    if intent == "role_transition" and state.get("current_role"):
        parts.append(f"Current role (to transition FROM): {state['current_role']}")
    if history_text := _format_history(state):
        parts.append(f"Recent conversation:\n{history_text}")
    parts.append(f"User request: {state['user_query']}")
    if state.get("metadata_filters"):
        parts.append(f"Extracted filters: {state['metadata_filters']}")

    context = "\n\n".join(parts)

    result: SearchQueries = await llm.ainvoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
    )

    queries = result.queries[:3]

    return {
        "search_queries": queries,
        "status_stream": state.get("status_stream", []) + [
            f"Generating {len(queries)} search queries…"
        ],
    }
