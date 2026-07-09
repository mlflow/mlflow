"""add endpoint_id to budget_policies

Adds an ``endpoint_id`` column to the ``budget_policies`` table so budget
policies can be scoped to a single gateway endpoint (target_scope=ENDPOINT).

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
        sa.Column("endpoint_id", sa.String(length=36), nullable=True),
    )
    op.create_index("idx_budget_policies_endpoint_id", "budget_policies", ["endpoint_id"])


def downgrade():
    op.drop_index("idx_budget_policies_endpoint_id", table_name="budget_policies")
    op.drop_column("budget_policies", "endpoint_id")
