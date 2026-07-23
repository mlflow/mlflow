"""add target_value to budget_policies

Adds a ``target_value`` column to the ``budget_policies`` table so budget
policies can be scoped to a single target, interpreted per ``target_scope``:
a gateway endpoint ID for ENDPOINT-scoped policies.

Create Date: 2026-07-09 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c3d9e7f1a2b4"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "budget_policies",
        sa.Column("target_value", sa.String(length=255), nullable=True),
    )
    op.create_index("idx_budget_policies_target_value", "budget_policies", ["target_value"])


def downgrade():
    op.drop_index("idx_budget_policies_target_value", table_name="budget_policies")
    op.drop_column("budget_policies", "target_value")
