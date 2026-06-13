"""add derivatives_snapshot table

Revision ID: a9f3c2b1d8e5
Revises: 35a41b884339
Create Date: 2026-06-13 19:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a9f3c2b1d8e5"
down_revision: Union[str, None] = "35a41b884339"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "derivatives_snapshot",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("funding_rate", sa.Float(), nullable=True),
        sa.Column("open_interest", sa.Float(), nullable=True),
        sa.Column("long_short_ratio", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_derivatives_snapshot_symbol_ts",
        "derivatives_snapshot",
        ["symbol", "ts"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_derivatives_snapshot_symbol_ts", table_name="derivatives_snapshot")
    op.drop_table("derivatives_snapshot")
