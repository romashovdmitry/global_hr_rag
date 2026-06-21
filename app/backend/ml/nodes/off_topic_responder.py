"""LangGraph node: respond to queries outside the job-vacancy domain."""

from backend.ml.state import RAGState

_RESPONSE = (
    "I'm a job vacancy assistant and can only help with IT career and job search questions. "
    "I can help you find matching vacancies, explore career transitions, or compare positions. "
    "What would you like to know?"
)


async def off_topic_responder_node(state: RAGState) -> dict:
    """Return a fixed response for off-topic queries, bypassing the full RAG pipeline.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``generated_answer`` and a status entry.
    """
    return {
        "generated_answer": _RESPONSE,
        "is_grounded": True,
        "status_stream": state.get("status_stream", []) + ["Off-topic query — skipping RAG pipeline."],
    }
