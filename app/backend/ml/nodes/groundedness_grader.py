"""LangGraph node: verify the generated answer is grounded in retrieved vacancy data."""

from pydantic import BaseModel
from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node


class GroundednessResult(BaseModel):
    """Structured output from the groundedness check."""

    is_grounded: bool
    reason: str


_SYSTEM = """You are a hallucination detector for a job-matching assistant.
Given vacancy source data and a generated answer, determine whether every factual claim
in the answer (salary figures, tech stacks, company names, benefits) is explicitly supported
by the source data.

Return JSON with:
- "is_grounded": true if the answer contains NO invented facts, false otherwise
- "reason": one-sentence explanation"""


@timed_node("groundedness_grader")
async def groundedness_grader_node(state: RAGState) -> dict:
    """Check whether the generated answer is grounded in the vacancy source data.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``is_grounded``, ``retry_count``, and status.
    """
    answer = state.get("generated_answer", "")
    vacancies = state.get("graded_vacancies", [])

    if not vacancies:
        return {
            "is_grounded": True,
            "retry_count": state.get("retry_count", 0),
            "status_stream": state.get("status_stream", []) + ["Groundedness: N/A (no vacancies)."],
        }

    source_block = "\n\n".join(
        f"Role: {v.get('role','')}, Company: {v.get('company','')}, "
        f"Salary: {v.get('salary')}, Stack: {v.get('description_of_vacancy','')[:400]}"
        for v in vacancies
    )

    llm = get_llm_service().get_llm().with_structured_output(GroundednessResult)

    result: GroundednessResult = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Source vacancy data:\n{source_block}\n\n"
                f"Generated answer:\n{answer}"
            ),
        },
    ])

    retry_count = state.get("retry_count", 0)
    if not result.is_grounded:
        retry_count += 1
        generated_answer = ""
        grounding_feedback = result.reason
        rejected_answer = answer
    else:
        generated_answer = answer
        grounding_feedback = None
        rejected_answer = None

    return {
        "is_grounded": result.is_grounded,
        "generated_answer": generated_answer,
        "grounding_feedback": grounding_feedback,
        "rejected_answer": rejected_answer,
        "retry_count": retry_count,
        "status_stream": state.get("status_stream", []) + [
            f"Groundedness check: {'✓ passed' if result.is_grounded else f'✗ failed — {result.reason}'}"
        ],
    }
