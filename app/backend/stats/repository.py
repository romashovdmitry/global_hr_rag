"""Data-access layer for query statistics."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.stats.models import QueryStat


async def save(db: AsyncSession, stat: QueryStat) -> None:
    """Persist a QueryStat row.

    Args:
        db: Active async database session.
        stat: Already-constructed QueryStat instance.
    """
    db.add(stat)
    await db.commit()


async def get_recent(db: AsyncSession, limit: int = 50) -> list[QueryStat]:
    """Return the most recent query stats rows.

    Args:
        db: Active async database session.
        limit: How many rows to return (default 50).

    Returns:
        List of QueryStat instances ordered by newest first.
    """
    result = await db.execute(
        select(QueryStat)
        .order_by(QueryStat.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_aggregates(db: AsyncSession) -> dict:
    """Compute aggregated statistics across all stored query rows.

    Returns:
        Dict with total_queries, avg_latency_ms, fallback_rate, off_topic_rate,
        avg_input_tokens, avg_output_tokens, intent_distribution,
        avg_node_latencies_ms.
    """
    total_result = await db.execute(select(func.count()).select_from(QueryStat))
    total = total_result.scalar_one() or 0

    if total == 0:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "fallback_rate": 0.0,
            "off_topic_rate": 0.0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "intent_distribution": {},
            "avg_node_latencies_ms": {},
        }

    rows = await get_recent(db, limit=500)

    avg_latency = sum(r.total_latency_ms for r in rows) / len(rows)
    fallback_count = sum(1 for r in rows if r.was_fallback)
    off_topic_count = sum(1 for r in rows if r.was_off_topic)

    tokens_in = [r.input_tokens for r in rows if r.input_tokens is not None]
    tokens_out = [r.output_tokens for r in rows if r.output_tokens is not None]

    intent_dist: dict[str, int] = {}
    for r in rows:
        intent_dist[r.intent] = intent_dist.get(r.intent, 0) + 1

    # Aggregate per-node latencies across all rows that have them.
    node_totals: dict[str, list[int]] = {}
    for r in rows:
        for node, ms in (r.node_timings_ms or {}).items():
            node_totals.setdefault(node, []).append(ms)
    avg_node = {node: int(sum(vals) / len(vals)) for node, vals in node_totals.items()}

    return {
        "total_queries": total,
        "avg_latency_ms": int(avg_latency),
        "fallback_rate": round(fallback_count / len(rows), 3),
        "off_topic_rate": round(off_topic_count / len(rows), 3),
        "avg_input_tokens": int(sum(tokens_in) / len(tokens_in)) if tokens_in else 0,
        "avg_output_tokens": int(sum(tokens_out) / len(tokens_out)) if tokens_out else 0,
        "intent_distribution": intent_dist,
        "avg_node_latencies_ms": avg_node,
    }
