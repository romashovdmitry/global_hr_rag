"""LangGraph node: classify the intent of the user's message."""

from pydantic import BaseModel

from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node


class IntentResult(BaseModel):
    """Structured output of the intent classifier."""

    intent: str
    current_role: str | None = None


_SYSTEM = """You are an intent classifier for a job-vacancy assistant.

Classify the user's message into exactly one of these intents:

- find_matching     : User wants to find vacancies that match their skills, CV, or requirements.
                      Examples: "show me Python jobs", "what vacancies suit me", "find remote roles".

- role_transition   : User explicitly wants to switch, change, or move AWAY from their current role
                      into a DIFFERENT one. The key signal is "switch", "change my role", "move to
                      another field", "I want to try something different", "looking for a career change".
                      Examples: "I'm a QA engineer and want to switch roles",
                                "I want to move from QA to backend development".

- compare           : User wants to compare specific vacancies OR pick the single best one from
                      already-presented options.
                      Examples: "which is better", "what's the best from these", "compare the two roles".

- clarify_vacancy   : User asks a follow-up question about a specific vacancy mentioned earlier.
                      Examples: "tell me more about the first one", "what's the salary at that company".

- followup          : User refines or continues the previous query without changing the core intent.
                      Examples: "and what about remote ones", "add salary filter above 4000".

- aggregate         : User asks a question that requires synthesising or comparing information
                      ACROSS MULTIPLE vacancies (more than 2–3), such as aggregations, statistics,
                      or filtering patterns across the whole database.
                      Examples: "which companies offer relocation AND health insurance",
                                "compare salary ranges for Middle vs Senior Python devs",
                                "what's the average salary for remote backend roles",
                                "list all companies that don't require English".

- off_topic         : The user's message has NOTHING to do with job searching, IT careers, or vacancies.
                      This includes messages in ANY language — Russian, Ukrainian, or others.
                      A short or informal message that is still about finding work, IT positions, or career
                      topics is NOT off_topic — classify it by its actual job-search intent instead.
                      Examples of off_topic: "how to cook pasta", "who is Superman", "capital of France",
                                             "tell me a joke", "what is 2+2".
                      Examples that are NOT off_topic (classify as find_matching or followup):
                        "есть что по мобилкам?" → find_matching
                        "ищу нормальную работу на питоне" → find_matching
                        "покажи удалённые позиции" → find_matching or followup

Return JSON with:
- "intent": one of the seven strings above.
- "current_role": if intent is "role_transition", the role the user is transitioning FROM
  (e.g. "QA engineer", "frontend developer"). Otherwise null."""


def _format_history(state: RAGState) -> str:
    history = state.get("chat_history") or []
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])


@timed_node("intent_classifier")
async def intent_classifier_node(state: RAGState) -> dict:
    """Classify the user's intent and extract any role-transition context.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``intent``, ``current_role``, and a status entry.
    """
    llm = get_llm_service().get_llm(temperature=0).with_structured_output(IntentResult)

    parts: list[str] = []
    if history := _format_history(state):
        parts.append(f"Recent conversation:\n{history}")
    parts.append(f"Latest user message: {state['user_query']}")
    context = "\n\n".join(parts)

    result: IntentResult = await llm.ainvoke(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": context}]
    )

    intent = result.intent if result.intent in (
        "find_matching", "role_transition", "compare", "clarify_vacancy",
        "followup", "aggregate", "off_topic"
    ) else "find_matching"

    return {
        "intent": intent,
        "current_role": result.current_role,
        "exclude_skills": [],
        "status_stream": state.get("status_stream", []) + [
            f"Intent: {intent}" + (f" (from: {result.current_role})" if result.current_role else "")
        ],
    }
