"""add trace tables

Revision ID: 867495a8f9d4
Revises: acf3f17fdcc7
Create Date: 2024-04-27 12:29:25.178685

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlTraceInfo, SqlTraceMetadata, SqlTraceTag

# revision identifiers, used by Alembic.
revision = "867495a8f9d4"
down_revision = "acf3f17fdcc7"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlTraceInfo.__tablename__,
        sa.Column("request_id", sa.String(length=50), primary_key=True, nullable=False),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey(
                column="experiments.experiment_id",
                name="fk_trace_info_experiment_id",
            ),
            nullable=False,
        ),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("execution_time_ms", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("request_id", name="trace_info_pk"),
        sa.Index(
            f"index_{SqlTraceInfo.__tablename__}_experiment_id_timestamp_ms",
            "experiment_id",
            "timestamp_ms",
            unique=False,
        ),
    )
    op.create_table(
        SqlTraceTag.__tablename__,
        sa.Column("key", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=8000), nullable=True),
        sa.Column(
            "request_id",
            sa.String(length=50),
            sa.ForeignKey(
                column=SqlTraceInfo.request_id,
                name=f"fk_{SqlTraceTag.__tablename__}_request_id",
            ),
            nullable=False,
            primary_key=True,
        ),
        sa.PrimaryKeyConstraint("key", "request_id", name="trace_tag_pk"),
        sa.Index(
            f"index_{SqlTraceTag.__tablename__}_request_id",
            "request_id",
            unique=False,
        ),
    )
    op.create_table(
        SqlTraceMetadata.__tablename__,
        sa.Column("key", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=8000), nullable=True),
        sa.Column(
            "request_id",
            sa.String(length=50),
            sa.ForeignKey(
                column=SqlTraceInfo.request_id,
                name=f"fk_{SqlTraceMetadata.__tablename__}_request_id",
            ),
            nullable=False,
            primary_key=True,
        ),
        sa.PrimaryKeyConstraint("key", "request_id", name="trace_request_metadata_pk"),
        sa.Index(
            f"index_{SqlTraceMetadata.__tablename__}_request_id",
            "request_id",
            unique=False,
        ),
    )


def downgrade():
    pass
