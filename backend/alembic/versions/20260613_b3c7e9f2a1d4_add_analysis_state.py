"""add analysis_state table

Revision ID: b3c7e9f2a1d4
Revises: a9f3c2b1d8e5
Create Date: 2026-06-13 22:30:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b3c7e9f2a1d4"
down_revision: str | None = "a9f3c2b1d8e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_state",
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=5), nullable=False),
        sa.Column("last_candle_ts", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "timeframe"),
    )


def downgrade() -> None:
    op.drop_table("analysis_state")
