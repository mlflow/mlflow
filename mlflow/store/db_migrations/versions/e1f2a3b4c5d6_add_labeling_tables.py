"""add labeling tables

Create Date: 2026-02-15 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e1f2a3b4c5d6"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade():
    # Create labeling_sessions table
    op.create_table(
        "labeling_sessions",
        sa.Column("labeling_session_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_labeling_sessions_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("labeling_session_id", name="labeling_session_pk"),
    )
    with op.batch_alter_table("labeling_sessions", schema=None) as batch_op:
        batch_op.create_index(
            "index_labeling_sessions_experiment_id",
            ["experiment_id"],
            unique=False,
        )

    # Create labeling_schemas table
    op.create_table(
        "labeling_schemas",
        sa.Column("labeling_schema_id", sa.String(length=36), nullable=False),
        sa.Column("labeling_session_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("assessment_type", sa.String(length=32), nullable=False),
        sa.Column("assessment_value_type", sa.Text(), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.Column("creation_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["labeling_session_id"],
            ["labeling_sessions.labeling_session_id"],
            name="fk_labeling_schemas_session_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("labeling_schema_id", name="labeling_schema_pk"),
    )
    with op.batch_alter_table("labeling_schemas", schema=None) as batch_op:
        batch_op.create_index(
            "index_labeling_schemas_session_id",
            ["labeling_session_id"],
            unique=False,
        )
        batch_op.create_index(
            "unique_labeling_schema_session_name",
            ["labeling_session_id", "name"],
            unique=True,
        )

    # Create labeling_session_items table
    op.create_table(
        "labeling_session_items",
        sa.Column("labeling_item_id", sa.String(length=36), nullable=False),
        sa.Column("labeling_session_id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=50), nullable=True),
        sa.Column("dataset_record_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'SKIPPED')",
            name="labeling_item_status",
        ),
        sa.ForeignKeyConstraint(
            ["labeling_session_id"],
            ["labeling_sessions.labeling_session_id"],
            name="fk_labeling_items_session_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("labeling_item_id", name="labeling_item_pk"),
    )
    with op.batch_alter_table("labeling_session_items", schema=None) as batch_op:
        batch_op.create_index(
            "index_labeling_items_session_id",
            ["labeling_session_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_labeling_items_trace_id",
            ["trace_id"],
            unique=False,
        )


def downgrade():
    op.drop_table("labeling_session_items")
    op.drop_table("labeling_schemas")
    op.drop_table("labeling_sessions")
