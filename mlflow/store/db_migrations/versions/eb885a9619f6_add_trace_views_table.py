"""add trace_views and trace_view_ranges tables

Create Date: 2026-03-26 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "eb885a9619f6"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "trace_views",
        sa.Column("view_id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("trace_id", sa.String(length=50), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_trace_views_trace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_trace_views_experiment_id",
        ),
        sa.PrimaryKeyConstraint("view_id", name="trace_views_pk"),
        sa.CheckConstraint(
            "(trace_id IS NOT NULL AND experiment_id IS NULL) OR "
            "(trace_id IS NULL AND experiment_id IS NOT NULL)",
            name="ck_trace_views_scope",
        ),
    )

    with op.batch_alter_table("trace_views", schema=None) as batch_op:
        batch_op.create_index(
            "index_trace_views_trace_id_created_timestamp",
            ["trace_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            "index_trace_views_experiment_id_created_timestamp",
            ["experiment_id", "created_timestamp"],
            unique=False,
        )

    op.create_table(
        "trace_view_ranges",
        sa.Column("range_id", sa.String(length=50), nullable=False),
        sa.Column("view_id", sa.String(length=50), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("from_selector", sa.Text(), nullable=False),
        sa.Column("to_selector", sa.Text(), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["view_id"],
            ["trace_views.view_id"],
            name="fk_trace_view_ranges_view_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("range_id", name="trace_view_ranges_pk"),
        sa.UniqueConstraint("view_id", "position", name="uq_trace_view_ranges_view_position"),
    )

    with op.batch_alter_table("trace_view_ranges", schema=None) as batch_op:
        batch_op.create_index(
            "index_trace_view_ranges_view_id_position",
            ["view_id", "position"],
            unique=False,
        )


def downgrade():
    op.drop_table("trace_view_ranges")
    op.drop_table("trace_views")
