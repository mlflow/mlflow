"""add_oauth_tables

Revision ID: b4a5d6e7f8g9
Revises: 2ed73881770d
Create Date: 2026-03-04 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b4a5d6e7f8g9"
down_revision = "2ed73881770d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), nullable=False, primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("access_token_enc", sa.Text(), nullable=True),
        sa.Column("refresh_token_enc", sa.Text(), nullable=True),
        sa.Column("id_token_claims", sa.Text(), nullable=True),
        sa.Column("token_expiry", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_sessions_user_id"),
    )
    op.create_index("idx_sessions_user_id", "sessions", ["user_id"])
    op.create_index("idx_sessions_expires_at", "sessions", ["expires_at"])

    op.create_table(
        "oauth_state",
        sa.Column("state", sa.String(length=64), nullable=False, primary_key=True),
        sa.Column("code_verifier", sa.String(length=128), nullable=True),
        sa.Column("nonce", sa.String(length=64), nullable=True),
        sa.Column("provider_name", sa.String(length=64), nullable=False),
        sa.Column("redirect_after_login", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "user_role_overrides",
        sa.Column("user_id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("default_permission", sa.String(length=32), nullable=False),
        sa.Column("idp_groups", sa.Text(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_role_overrides_user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_role_overrides")
    op.drop_table("oauth_state")
    op.drop_index("idx_sessions_expires_at", table_name="sessions")
    op.drop_index("idx_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
