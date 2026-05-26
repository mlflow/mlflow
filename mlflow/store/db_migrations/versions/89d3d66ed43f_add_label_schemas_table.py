"""add label_schemas table

Revision ID: 89d3d66ed43f
Revises: da6fb0208061

Create Date: 2026-05-26 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# revision identifiers, used by Alembic.
revision = "89d3d66ed43f"
down_revision = "da6fb0208061"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "label_schemas",
        sa.Column("schema_id", sa.String(length=36), nullable=False),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text(f"'{DEFAULT_WORKSPACE_NAME}'"),
        ),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=True),
        sa.Column("enable_comment", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("input_type", sa.String(length=32), nullable=False),
        sa.Column("input_config", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_label_schemas_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("schema_id", name="label_schemas_pk"),
        sa.UniqueConstraint(
            "workspace",
            "experiment_id",
            "name",
            name="uq_label_schemas_workspace_exp_name",
        ),
    )
    op.create_index(
        "index_label_schemas_workspace_experiment_id",
        "label_schemas",
        ["workspace", "experiment_id"],
        unique=False,
    )


def downgrade():
    op.drop_index("index_label_schemas_workspace_experiment_id", table_name="label_schemas")
    op.drop_table("label_schemas")
