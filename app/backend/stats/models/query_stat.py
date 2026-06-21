"""SQLAlchemy model for per-query RAG pipeline statistics."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class QueryStat(Base):
    """Stores one row per user query with latency, intent, and quality signals.

    Collected automatically in ``chat/service.py`` after each pipeline run.
    Exposes raw data for the ``/stats`` dashboard.
    """

    __tablename__ = "query_stats"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    user_query: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(50), index=True)
    total_latency_ms: Mapped[int] = mapped_column(Integer)
    node_timings_ms: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieved_count: Mapped[int] = mapped_column(Integer, default=0)
    graded_count: Mapped[int] = mapped_column(Integer, default=0)
    is_grounded: Mapped[bool] = mapped_column(Boolean, default=True)
    was_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    was_off_topic: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
