"""Stats API — exposes query-level performance and quality metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_db
from backend.stats import repository
from backend.stats.models import QueryStat
from backend.stats.schemas import QueryStatRead, StatsAggregatesResponse

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get(
    "/",
    response_model=StatsAggregatesResponse,
    summary="Aggregated RAG pipeline metrics",
    description=(
        "Returns aggregated latency, fallback, off-topic, token, intent, "
        "and per-node timing metrics for completed RAG queries."
    ),
)
async def get_stats(db: AsyncSession = Depends(get_db)) -> StatsAggregatesResponse:
    """Return aggregated statistics across all recorded queries.

    Includes intent distribution, average latency per pipeline node,
    fallback rate, off-topic rate, and token usage.
    """
    return StatsAggregatesResponse.model_validate(await repository.get_aggregates(db))


@router.delete("/", summary="Clear all stats", status_code=204)
async def clear_stats(db: AsyncSession = Depends(get_db)) -> None:
    """Delete all rows from the query_stats table.

    Args:
        db: Active async database session.
    """
    await db.execute(delete(QueryStat))
    await db.commit()


@router.get(
    "/recent",
    response_model=list[QueryStatRead],
    summary="Recent query log",
    description="Returns the most recent query-stat rows ordered newest-first.",
)
async def get_recent(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[QueryStatRead]:
    """Return the most recent query stat rows.

    Args:
        limit: Number of rows to return (1–200).

    Returns:
        List of query stat dicts ordered newest-first.
    """
    rows: list[QueryStat] = await repository.get_recent(db, limit=limit)
    return [
        QueryStatRead(
            id=r.id,
            timestamp=r.created_at.isoformat(),
            user_query=r.user_query,
            intent=r.intent,
            total_latency_ms=r.total_latency_ms,
            node_timings_ms=r.node_timings_ms,
            retrieved_count=r.retrieved_count,
            graded_count=r.graded_count,
            is_grounded=r.is_grounded,
            was_fallback=r.was_fallback,
            was_off_topic=r.was_off_topic,
            retry_count=r.retry_count,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
        )
        for r in rows
    ]
