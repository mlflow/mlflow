"""create entity_associations table

Revision ID: de4033877273
Revises: a1b2c3d4e5f6
Create Date: 2025-07-28 13:05:53.982327

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "de4033877273"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade():
    # Create entity_associations table
    op.create_table(
        "entity_associations",
        sa.Column("association_id", sa.String(36), nullable=False),
        sa.Column("source_type", sa.String(36), nullable=False),
        sa.Column("source_id", sa.String(36), nullable=False),
        sa.Column("destination_type", sa.String(36), nullable=False),
        sa.Column("destination_id", sa.String(36), nullable=False),
        sa.Column("created_time", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint(
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
            name="entity_associations_pk",
        ),
    )

    # Create indexes on entity_associations
    with op.batch_alter_table("entity_associations", schema=None) as batch_op:
        batch_op.create_index(
            "index_entity_associations_association_id",
            ["association_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_entity_associations_reverse_lookup",
            ["destination_type", "destination_id", "source_type", "source_id"],
            unique=False,
        )


def downgrade():
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_table("entity_associations")
