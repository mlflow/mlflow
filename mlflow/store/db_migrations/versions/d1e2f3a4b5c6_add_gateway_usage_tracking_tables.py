"""add gateway usage tracking tables

Create Date: 2025-01-09 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d1e2f3a4b5c6"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    # Create gateway_invocations table
    op.create_table(
        "gateway_invocations",
        sa.Column("invocation_id", sa.String(length=36), nullable=False),
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("endpoint_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("total_prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_latency_ms", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("invocation_id", name="gateway_invocations_pk"),
    )

    # Create indexes for gateway_invocations
    with op.batch_alter_table("gateway_invocations", schema=None) as batch_op:
        batch_op.create_index(
            "index_gateway_invocations_endpoint_id",
            ["endpoint_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_invocations_created_at",
            ["created_at"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_invocations_status",
            ["status"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_invocations_username",
            ["username"],
            unique=False,
        )

    # Create gateway_provider_calls table
    op.create_table(
        "gateway_provider_calls",
        sa.Column("provider_call_id", sa.String(length=36), nullable=False),
        sa.Column("invocation_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("completion_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["invocation_id"],
            ["gateway_invocations.invocation_id"],
            name="fk_provider_calls_invocation_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("provider_call_id", name="gateway_provider_calls_pk"),
    )

    # Create indexes for gateway_provider_calls
    with op.batch_alter_table("gateway_provider_calls", schema=None) as batch_op:
        batch_op.create_index(
            "index_gateway_provider_calls_invocation_id",
            ["invocation_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_provider_calls_provider",
            ["provider"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_provider_calls_model_name",
            ["model_name"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_provider_calls_created_at",
            ["created_at"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_provider_calls_status",
            ["status"],
            unique=False,
        )


def downgrade():
    op.drop_table("gateway_provider_calls")
    op.drop_table("gateway_invocations")
