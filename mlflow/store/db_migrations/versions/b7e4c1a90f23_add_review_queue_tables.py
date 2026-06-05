"""add review_queue tables

Revision ID: b7e4c1a90f23
Revises: 89d3d66ed43f

Create Date: 2026-06-05 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b7e4c1a90f23"
down_revision = "89d3d66ed43f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "review_queues",
        sa.Column("queue_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=250), nullable=False),
        sa.Column("queue_type", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("creation_time_ms", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time_ms", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_review_queues_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("queue_id", name="review_queues_pk"),
        sa.UniqueConstraint(
            "experiment_id",
            "name",
            name="uq_review_queues_experiment_name",
        ),
    )
    op.create_index(
        "index_review_queues_experiment_id",
        "review_queues",
        ["experiment_id"],
        unique=False,
    )

    op.create_table(
        "review_queue_users",
        sa.Column("queue_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=250), nullable=False),
        sa.ForeignKeyConstraint(
            ["queue_id"],
            ["review_queues.queue_id"],
            name="fk_review_queue_users_queue_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("queue_id", "user_id", name="review_queue_users_pk"),
    )
    op.create_index(
        "index_review_queue_users_user_id",
        "review_queue_users",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "review_queue_traces",
        sa.Column("queue_id", sa.String(length=36), nullable=False),
        sa.Column("target_type", sa.String(length=16), nullable=False),
        sa.Column("target_id", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("completed_by", sa.String(length=250), nullable=True),
        sa.Column("completed_time_ms", sa.BigInteger(), nullable=True),
        sa.Column("creation_time_ms", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time_ms", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["queue_id"],
            ["review_queues.queue_id"],
            name="fk_review_queue_traces_queue_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("queue_id", "target_id", name="review_queue_traces_pk"),
    )
    op.create_index(
        "index_review_queue_traces_queue_id_status",
        "review_queue_traces",
        ["queue_id", "status"],
        unique=False,
    )
    op.create_index(
        "index_review_queue_traces_target_id",
        "review_queue_traces",
        ["target_id"],
        unique=False,
    )

    op.create_table(
        "review_queue_label_schemas",
        sa.Column("queue_id", sa.String(length=36), nullable=False),
        sa.Column("schema_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["queue_id"],
            ["review_queues.queue_id"],
            name="fk_review_queue_label_schemas_queue_id",
            ondelete="CASCADE",
        ),
        # `schema_id` is intentionally NOT a foreign key: a second cascading
        # FK to `label_schemas` would converge with the `queue_id` ->
        # `review_queues` -> `experiments` cascade on an experiment delete,
        # which MSSQL rejects as a multiple-cascade-path. Existence is
        # enforced at write time in the store instead.
        sa.PrimaryKeyConstraint("queue_id", "schema_id", name="review_queue_label_schemas_pk"),
    )
    op.create_index(
        "index_review_queue_label_schemas_schema_id",
        "review_queue_label_schemas",
        ["schema_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "index_review_queue_label_schemas_schema_id",
        table_name="review_queue_label_schemas",
    )
    op.drop_table("review_queue_label_schemas")

    op.drop_index(
        "index_review_queue_traces_target_id",
        table_name="review_queue_traces",
    )
    op.drop_index(
        "index_review_queue_traces_queue_id_status",
        table_name="review_queue_traces",
    )
    op.drop_table("review_queue_traces")

    op.drop_index("index_review_queue_users_user_id", table_name="review_queue_users")
    op.drop_table("review_queue_users")

    op.drop_index("index_review_queues_experiment_id", table_name="review_queues")
    op.drop_table("review_queues")
