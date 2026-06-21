"""Pydantic schemas for RAG pipeline statistics API responses."""

from pydantic import BaseModel


class StatsAggregatesResponse(BaseModel):
    """Aggregated RAG performance and quality metrics."""

    total_queries: int
    avg_latency_ms: int
    fallback_rate: float
    off_topic_rate: float
    avg_input_tokens: int
    avg_output_tokens: int
    intent_distribution: dict[str, int]
    avg_node_latencies_ms: dict[str, int]


class QueryStatRead(BaseModel):
    """Serialized query-stat row returned by the recent stats endpoint."""

    id: str
    timestamp: str
    user_query: str
    intent: str
    total_latency_ms: int
    node_timings_ms: dict[str, int]
    retrieved_count: int
    graded_count: int
    is_grounded: bool
    was_fallback: bool
    was_off_topic: bool
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
