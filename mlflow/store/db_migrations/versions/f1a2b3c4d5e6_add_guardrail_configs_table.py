"""add guardrail_configs table

Create Date: 2026-03-13 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "76601a5f987d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "guardrail_configs",
        sa.Column("guardrail_id", sa.String(length=36), nullable=False),
        sa.Column("endpoint_name", sa.String(length=255), nullable=True),
        sa.Column("scorer_name", sa.String(length=255), nullable=False),
        sa.Column("hook", sa.String(length=32), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.PrimaryKeyConstraint("guardrail_id", name="guardrail_configs_pk"),
    )
    op.create_index("idx_guardrail_configs_endpoint", "guardrail_configs", ["endpoint_name"])
    op.create_index("idx_guardrail_configs_workspace", "guardrail_configs", ["workspace"])


def downgrade():
    op.drop_index("idx_guardrail_configs_workspace", table_name="guardrail_configs")
    op.drop_index("idx_guardrail_configs_endpoint", table_name="guardrail_configs")
    op.drop_table("guardrail_configs")
