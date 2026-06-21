"""LangGraph RAG pipeline graph assembly."""

from langgraph.graph import END, START, StateGraph

from backend.ml.nodes.generator import generator_node
from backend.ml.nodes.groundedness_grader import groundedness_grader_node
from backend.ml.nodes.intent_classifier import intent_classifier_node
from backend.ml.nodes.metadata_extractor import metadata_extractor_node
from backend.ml.nodes.off_topic_responder import off_topic_responder_node
from backend.ml.nodes.query_translator import query_translator_node
from backend.ml.nodes.relevance_grader import relevance_grader_node
from backend.ml.nodes.retriever import retriever_node
from backend.ml.state import RAGState
from backend.core.config import settings


def _route_after_groundedness(state: RAGState) -> str:
    """Decide whether to end or re-route after the groundedness check.

    Retries up to ``settings.rag_max_retries`` times by going back to
    ``query_translator`` to produce different search queries.

    Args:
        state: Current pipeline state.

    Returns:
        ``"end"`` or ``"retry"``.
    """
    if state.get("is_grounded", True):
        return "end"
    if state.get("retry_count", 0) >= settings.rag_max_retries:
        return "end"
    return "retry"


def build_graph() -> StateGraph:
    """Assemble and compile the RAG pipeline graph.

    Returns:
        Compiled LangGraph ``StateGraph`` ready to invoke.
    """
    builder = StateGraph(RAGState)

    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("off_topic_responder", off_topic_responder_node)
    builder.add_node("metadata_extractor", metadata_extractor_node)
    builder.add_node("query_translator", query_translator_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("relevance_grader", relevance_grader_node)
    builder.add_node("generator", generator_node)
    builder.add_node("groundedness_grader", groundedness_grader_node)

    builder.add_edge(START, "intent_classifier")
    builder.add_conditional_edges(
        "intent_classifier",
        lambda s: "off_topic" if s.get("intent") == "off_topic" else "rag",
        {"off_topic": "off_topic_responder", "rag": "metadata_extractor"},
    )
    # aggregate uses the same pipeline but retriever.py automatically widens top_k.
    builder.add_edge("off_topic_responder", END)
    builder.add_edge("metadata_extractor", "query_translator")
    builder.add_edge("query_translator", "retriever")
    builder.add_edge("retriever", "relevance_grader")
    builder.add_edge("relevance_grader", "generator")
    builder.add_edge("generator", "groundedness_grader")

    builder.add_conditional_edges(
        "groundedness_grader",
        _route_after_groundedness,
        {"end": END, "retry": "query_translator"},
    )

    return builder.compile()


rag_graph = build_graph()
