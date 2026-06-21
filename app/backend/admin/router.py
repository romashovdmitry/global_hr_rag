"""Read-only admin endpoints for inspecting PostgreSQL and Qdrant state.

These routes are intentionally read-only and scoped to development/debugging.
They are auto-exposed as MCP tools by ``fastapi-mcp`` so Cursor agents can
inspect application state without needing direct database access.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.chat.models.session import Session
from backend.chat.models.message import Message
from backend.core.config import settings
from backend.db.session import get_db
from backend.utils.qdrant import embed_dense, embed_sparse_async, get_qdrant_client
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin (read-only)"])


class SessionSummary(BaseModel):
    """Lightweight session representation for admin listing."""

    id: str
    has_cv: bool
    message_count: int
    summary_length: int
    summarized_through: int


class QdrantCollectionStats(BaseModel):
    """Key statistics about the Qdrant vacancy collection."""

    collection_name: str
    points_count: int
    vectors_count: int
    status: str


class VacancyHit(BaseModel):
    """Single result from a direct Qdrant search."""

    id: str
    score: float
    role: str | None
    company: str | None
    salary: int | None
    status_of_role: list[str]


# PostgreSQL read endpoints

@router.get(
    "/sessions",
    response_model=list[SessionSummary],
    summary="List recent chat sessions",
    description=(
        "Returns the most recent chat sessions ordered by creation time. "
        "Includes per-session message count and rolling summary status."
    ),
)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[SessionSummary]:
    """Return the most recent sessions with lightweight statistics.

    Args:
        limit: Maximum number of sessions to return (1–100).
        db: Active async database session.

    Returns:
        List of SessionSummary objects ordered by creation time descending.
    """
    result = await db.execute(
        select(Session).order_by(Session.created_at.desc()).limit(limit)
    )
    sessions = result.scalars().all()

    summaries: list[SessionSummary] = []
    for s in sessions:
        msg_count_result = await db.execute(
            select(func.count()).where(Message.session_id == s.id)
        )
        msg_count = msg_count_result.scalar_one()
        summaries.append(
            SessionSummary(
                id=s.id,
                has_cv=s.cv_text is not None,
                message_count=msg_count,
                summary_length=len(s.summary) if s.summary else 0,
                summarized_through=s.summarized_through,
            )
        )
    return summaries


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[dict[str, Any]],
    summary="List all messages in a session",
    description="Returns every message in the session ordered chronologically.",
)
async def list_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return all messages for a given session.

    Args:
        session_id: Target session UUID.
        db: Active async database session.

    Returns:
        List of message dicts with role, content, and created_at.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]


# Qdrant read endpoints

@router.get(
    "/qdrant/stats",
    response_model=QdrantCollectionStats,
    summary="Get Qdrant collection statistics",
    description=(
        "Returns current point count, vector count, and collection status "
        "for the vacancy collection."
    ),
)
async def qdrant_stats() -> QdrantCollectionStats:
    """Fetch Qdrant collection metadata.

    Returns:
        QdrantCollectionStats with current counts and status.

    Raises:
        HTTPException: 503 if Qdrant is unreachable.
    """
    client = get_qdrant_client()
    try:
        info = await client.get_collection(settings.qdrant_collection)
        # vectors_count was removed in qdrant-client 1.13+ in some response shapes.
        vectors_count = getattr(info, "vectors_count", None) or 0
        return QdrantCollectionStats(
            collection_name=settings.qdrant_collection,
            points_count=info.points_count or 0,
            vectors_count=vectors_count,
            status=str(info.status),
        )
    except Exception as exc:
        logger.exception("Failed to fetch Qdrant stats")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unavailable: {exc}",
        ) from exc
    finally:
        await client.close()


@router.get(
    "/qdrant/search",
    response_model=list[VacancyHit],
    summary="Direct hybrid search in Qdrant",
    description=(
        "Runs a hybrid dense+sparse search on the vacancy collection and returns "
        "raw scored hits. Useful for inspecting retrieval quality without the "
        "full RAG pipeline."
    ),
)
async def qdrant_search(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[VacancyHit]:
    """Search the Qdrant vacancy store directly.

    Args:
        query: Natural language query string.
        limit: Maximum results to return (1–50).

    Returns:
        List of VacancyHit objects ranked by combined dense+sparse score.

    Raises:
        HTTPException: 503 if Qdrant or embedding generation fails.
    """
    client = get_qdrant_client()
    try:
        dense_vec = (await embed_dense([query]))[0]
        sparse_vec = (await embed_sparse_async([query]))[0]

        dense_resp = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense_vec,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        sparse_resp = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using="sparse",
            limit=limit,
            with_payload=True,
        )

        seen: dict[str, VacancyHit] = {}
        for hit in dense_resp.points + sparse_resp.points:
            hit_id = str(hit.id)
            if hit_id not in seen:
                payload = hit.payload or {}
                seen[hit_id] = VacancyHit(
                    id=hit_id,
                    score=hit.score,
                    role=payload.get("role"),
                    company=payload.get("company"),
                    salary=payload.get("salary"),
                    status_of_role=payload.get("status_of_role") or [],
                )

        return list(seen.values())[:limit]
    except Exception as exc:
        logger.exception("Qdrant direct search failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {exc}",
        ) from exc
    finally:
        await client.close()
