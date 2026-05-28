"""add review_assignments table

Create Date: 2026-05-28 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c4d2e8a16f93"
down_revision = "da6fb0208061"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "review_assignments",
        sa.Column("assignment_id", sa.String(length=36), nullable=False),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(length=16), nullable=False),
        sa.Column("target_id", sa.String(length=50), nullable=False),
        sa.Column("reviewer", sa.String(length=250), nullable=False),
        sa.Column("assigner", sa.String(length=250), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("creation_time_ms", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time_ms", sa.BigInteger(), nullable=False),
        sa.Column("completed_time_ms", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("assignment_id", name="review_assignments_pk"),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_review_assignments_experiment_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "workspace",
            "target_id",
            "reviewer",
            name="uq_review_assignments_workspace_target_reviewer",
        ),
    )
    # "My assignments" query (Reviews page tabs + list_my_assignments SDK).
    op.create_index(
        "idx_review_assignments_workspace_experiment_reviewer_state",
        "review_assignments",
        ["workspace", "experiment_id", "reviewer", "state"],
    )
    # "Who's reviewing this trace?" query (per-trace assignees widget).
    op.create_index(
        "idx_review_assignments_workspace_target_id",
        "review_assignments",
        ["workspace", "target_id"],
    )


def downgrade():
    op.drop_index(
        "idx_review_assignments_workspace_target_id",
        table_name="review_assignments",
    )
    op.drop_index(
        "idx_review_assignments_workspace_experiment_reviewer_state",
        table_name="review_assignments",
    )
    op.drop_table("review_assignments")
