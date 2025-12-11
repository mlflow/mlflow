"""add logged model tables

Create Date: 2025-02-06 22:05:35.542613

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "400f98739977"
down_revision = "0584bdc529eb"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "logged_models",
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("artifact_location", sa.String(length=1000), nullable=False),
        sa.Column("creation_timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("lifecycle_stage", sa.String(length=32), nullable=True),
        sa.Column("model_type", sa.String(length=500), nullable=True),
        sa.Column("source_run_id", sa.String(length=32), nullable=True),
        sa.Column("status_message", sa.String(length=1000), nullable=True),
        sa.CheckConstraint(
            "lifecycle_stage IN ('active', 'deleted')", name="logged_models_lifecycle_stage_check"
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_models_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model_id", name="logged_models_pk"),
    )
    op.create_table(
        "logged_model_metrics",
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("metric_name", sa.String(length=500), nullable=False),
        sa.Column("metric_timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("metric_step", sa.BigInteger(), nullable=False),
        sa.Column("metric_value", sa.Float(precision=53), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("dataset_uuid", sa.String(length=36), nullable=True),
        sa.Column("dataset_name", sa.String(length=500), nullable=True),
        sa.Column("dataset_digest", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_metrics_experiment_id",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            name="fk_logged_model_metrics_model_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_uuid"], name="fk_logged_model_metrics_run_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint(
            "model_id",
            "metric_name",
            "metric_timestamp_ms",
            "metric_step",
            "run_id",
            name="logged_model_metrics_pk",
        ),
    )
    with op.batch_alter_table("logged_model_metrics", schema=None) as batch_op:
        batch_op.create_index("index_logged_model_metrics_model_id", ["model_id"], unique=False)

    op.create_table(
        "logged_model_params",
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("param_key", sa.String(length=255), nullable=False),
        sa.Column("param_value", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_params_experiment_id",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            name="fk_logged_model_params_model_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model_id", "param_key", name="logged_model_params_pk"),
    )
    op.create_table(
        "logged_model_tags",
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("tag_key", sa.String(length=255), nullable=False),
        sa.Column("tag_value", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_tags_experiment_id",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            name="fk_logged_model_tags_model_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model_id", "tag_key", name="logged_model_tags_pk"),
    )


def downgrade():
    pass
