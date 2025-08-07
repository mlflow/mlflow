"""add spans table

Revision ID: a1b2c3d4e5f6
Revises: 770bee3ae1dd
Create Date: 2025-08-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import LONGTEXT

from mlflow.store.tracking.dbmodels.models import SqlSpan

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "770bee3ae1dd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlSpan.__tablename__,
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("parent_span_id", sa.String(length=50), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        # Use String instead of Text for type column to support MSSQL indexes.
        # MSSQL doesn't allow TEXT columns in indexes. Limited to 500 chars
        # to stay within MySQL's max index key length of 3072 bytes.
        sa.Column("type", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("start_time_unix_nano", sa.BigInteger(), nullable=False),
        sa.Column("end_time_unix_nano", sa.BigInteger(), nullable=True),
        sa.Column(
            "duration_ns",
            sa.BigInteger(),
            sa.Computed("end_time_unix_nano - start_time_unix_nano", persisted=True),
            nullable=True,
        ),
        # Use LONGTEXT for MySQL to support large span content (up to 4GB).
        # Standard TEXT in MySQL is limited to 64KB which is insufficient for
        # spans with extensive attributes, events, or nested data structures.
        sa.Column("content", sa.Text().with_variant(LONGTEXT, "mysql"), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_spans_trace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_spans_experiment_id",
        ),
        sa.PrimaryKeyConstraint("trace_id", "span_id", name="spans_pk"),
    )

    with op.batch_alter_table(SqlSpan.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        # Two indexes needed to support both filter patterns efficiently:
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_status_type",
            ["experiment_id", "status", "type"],
            unique=False,
        )  # For status-only and status+type filters
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_type_status",
            ["experiment_id", "type", "status"],
            unique=False,
        )  # For type-only and type+status filters
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_duration",
            ["experiment_id", "duration_ns"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlSpan.__tablename__)
