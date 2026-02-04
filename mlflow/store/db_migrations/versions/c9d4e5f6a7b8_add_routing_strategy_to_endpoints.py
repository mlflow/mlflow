"""add routing strategy to endpoints and linkage type to mappings

Create Date: 2025-12-18 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c9d4e5f6a7b8"
down_revision = "5d2d30f0abce"
branch_labels = None
depends_on = None


def upgrade():
    # Add routing strategy and fallback config to endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("routing_strategy", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("fallback_config_json", sa.Text(), nullable=True))

    # Add linkage_type and fallback_order to endpoint_model_mappings table
    with op.batch_alter_table("endpoint_model_mappings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "linkage_type", sa.String(length=64), nullable=False, server_default="PRIMARY"
            )
        )
        batch_op.add_column(sa.Column("fallback_order", sa.Integer(), nullable=True))
        batch_op.drop_index("unique_endpoint_model_mapping")
        batch_op.create_index(
            "unique_endpoint_model_linkage_mapping",
            ["endpoint_id", "model_definition_id", "linkage_type"],
            unique=True,
        )


def downgrade():
    # Remove linkage_type and fallback_order from endpoint_model_mappings table
    with op.batch_alter_table("endpoint_model_mappings", schema=None) as batch_op:
        batch_op.drop_index("unique_endpoint_model_linkage_mapping")
        batch_op.create_index(
            "unique_endpoint_model_mapping",
            ["endpoint_id", "model_definition_id"],
            unique=True,
        )
        batch_op.drop_column("fallback_order")
        batch_op.drop_column("linkage_type")

    # Remove routing strategy and fallback config from endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_column("fallback_config_json")
        batch_op.drop_column("routing_strategy")
