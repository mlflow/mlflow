"""add_sessions_table

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
Create Date: 2026-09-08 00:00:00.000000

Adds the ``sessions`` table backing optional server-side login sessions
(``mlflow.server.auth.session:authenticate_request_session``). Not used by
the default ``authenticate_request_basic_auth`` authorization function, so
this table stays empty for deployments that don't opt in.

See https://github.com/mlflow/mlflow/issues/13643.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=255), nullable=False, primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_sessions_user_id"),
    )
    op.create_index("idx_sessions_user_id", "sessions", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
