"""add gateway rate limits table

Create Date: 2025-01-09 14:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e2f3a4b5c6d7"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade():
    # Create gateway_rate_limits table
    op.create_table(
        "gateway_rate_limits",
        sa.Column("rate_limit_id", sa.String(length=36), nullable=False),
        sa.Column("endpoint_id", sa.String(length=36), nullable=False),
        sa.Column("queries_per_minute", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("updated_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["endpoints.endpoint_id"],
            name="fk_rate_limits_endpoint_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("rate_limit_id", name="gateway_rate_limits_pk"),
        sa.UniqueConstraint("endpoint_id", "username", name="uq_rate_limit_endpoint_user"),
    )

    # Create indexes for gateway_rate_limits
    with op.batch_alter_table("gateway_rate_limits", schema=None) as batch_op:
        batch_op.create_index(
            "index_gateway_rate_limits_endpoint_id",
            ["endpoint_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_gateway_rate_limits_username",
            ["username"],
            unique=False,
        )


def downgrade():
    op.drop_table("gateway_rate_limits")
