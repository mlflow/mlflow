"""add principal column to budget_policies

Adds a ``principal`` column to ``budget_policies`` so budget policies can be
scoped to an individual user (USER target scope) in addition to the existing
GLOBAL, WORKSPACE, and ENDPOINT scopes.

Create Date: 2026-07-09 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c3d7f2a1b8e4"
down_revision = "c3d9e7f1a2b4"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "budget_policies",
        sa.Column("principal", sa.String(length=255), nullable=True),
    )
    op.create_index("idx_budget_policies_principal", "budget_policies", ["principal"])


def downgrade():
    op.drop_index("idx_budget_policies_principal", table_name="budget_policies")
    op.drop_column("budget_policies", "principal")
