"""Add mutation_conditions table for condition-based access control

Revision ID: b7c8d9e0f1a2
Revises: f1a2b3c4d5e6
Create Date: 2026-09-25 00:00:00.000000

Adds the ``mutation_conditions`` table: two optional filters per
``(role, resource_type, resource_pattern)`` that gate create/mutation operations
only. ``resource_pattern`` addresses the governed resources the way a grant does
-- ``"*"`` or one id -- and defaults to ``"*"``.

The table starts empty, and an empty table is exactly the pre-existing
behaviour -- conditions only ever subtract from what grants allow. So this
migration is behaviour-neutral on upgrade, and ``downgrade`` is lossy only in
that it drops conditions an admin authored after upgrading (it cannot grant
anyone access they did not already have).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b7c8d9e0f1a2"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mutation_conditions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "role_id",
            sa.Integer(),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        # Which resources of that type: "*" or a single id, at the grain the type's grants
        # use. NOT NULL with a wildcard default so that "every resource" has exactly one
        # representation -- a nullable column would let NULL and "*" both mean all.
        sa.Column("resource_pattern", sa.String(length=255), nullable=False, server_default="*"),
        sa.Column("value_condition", sa.Text(), nullable=True),
        sa.Column("target_condition", sa.Text(), nullable=True),
        sa.UniqueConstraint("role_id", "resource_type", name="unique_role_resource_type"),
    )
    op.create_index(
        "idx_mutation_conditions_role_id", "mutation_conditions", ["role_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_mutation_conditions_role_id", table_name="mutation_conditions")
    op.drop_table("mutation_conditions")
