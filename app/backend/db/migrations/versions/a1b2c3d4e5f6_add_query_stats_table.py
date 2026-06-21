"""Add query_stats table for per-query RAG pipeline metrics.

Revision ID: a1b2c3d4e5f6
Revises: e79fd9002acd
Create Date: 2026-06-21 16:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "e79fd9002acd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_stats",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(50), nullable=False),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False),
        sa.Column("node_timings_ms", sa.JSON(), nullable=True),
        sa.Column("retrieved_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("graded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_grounded", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("was_fallback", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("was_off_topic", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_stats_created_at", "query_stats", ["created_at"])
    op.create_index("ix_query_stats_intent", "query_stats", ["intent"])


def downgrade() -> None:
    op.drop_index("ix_query_stats_intent", table_name="query_stats")
    op.drop_index("ix_query_stats_created_at", table_name="query_stats")
    op.drop_table("query_stats")
