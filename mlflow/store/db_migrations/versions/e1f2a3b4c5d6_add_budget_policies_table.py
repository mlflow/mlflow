"""add budget_policies table

Create Date: 2026-02-24 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e1f2a3b4c5d6"
down_revision = "1b5f0d9ad7c1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "budget_policies",
        sa.Column("budget_policy_id", sa.String(length=36), nullable=False),
        sa.Column("budget_type", sa.String(length=32), nullable=False),
        sa.Column("budget_amount", sa.Float(), nullable=False),
        sa.Column("duration_type", sa.String(length=32), nullable=False),
        sa.Column("duration_value", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("on_exceeded", sa.String(length=32), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.PrimaryKeyConstraint("budget_policy_id", name="budget_policies_pk"),
    )
    op.create_index("idx_budget_policies_workspace", "budget_policies", ["workspace"])


def downgrade():
    op.drop_index("idx_budget_policies_workspace", table_name="budget_policies")
    op.drop_table("budget_policies")
