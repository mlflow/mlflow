"""add guardrails and guardrail_configs tables

Create Date: 2026-03-24 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "ae8bbe7743c9"
down_revision = "a5b4c3d2e1f0"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "guardrails",
        sa.Column("guardrail_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_version", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("action_endpoint_id", sa.String(length=36), nullable=True),
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
        sa.PrimaryKeyConstraint("guardrail_id", name="guardrails_pk"),
        sa.ForeignKeyConstraint(
            ["scorer_id", "scorer_version"],
            ["scorer_versions.scorer_id", "scorer_versions.scorer_version"],
            name="fk_guardrails_scorer_version",
        ),
        sa.ForeignKeyConstraint(
            ["action_endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_guardrails_action_endpoint_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("idx_guardrails_workspace", "guardrails", ["workspace"])
    op.create_index("idx_guardrails_scorer", "guardrails", ["scorer_id", "scorer_version"])

    op.create_table(
        "guardrail_configs",
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("guardrail_id", sa.String(length=36), nullable=False),
        sa.Column("execution_order", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.PrimaryKeyConstraint("endpoint_id", "guardrail_id", name="guardrail_configs_pk"),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_guardrail_configs_endpoint_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["guardrail_id"],
            ["guardrails.guardrail_id"],
            name="fk_guardrail_configs_guardrail_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_guardrail_configs_endpoint_id", "guardrail_configs", ["endpoint_id"])
    op.create_index("idx_guardrail_configs_guardrail_id", "guardrail_configs", ["guardrail_id"])


def downgrade():
    op.drop_index("idx_guardrail_configs_guardrail_id", table_name="guardrail_configs")
    op.drop_index("idx_guardrail_configs_endpoint_id", table_name="guardrail_configs")
    op.drop_table("guardrail_configs")
    op.drop_index("idx_guardrails_scorer", table_name="guardrails")
    op.drop_index("idx_guardrails_workspace", table_name="guardrails")
    op.drop_table("guardrails")
