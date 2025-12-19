"""add assessments table

Create Date: 2025-06-23 11:26:19.855639

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlAssessments

# revision identifiers, used by Alembic.
revision = "770bee3ae1dd"
down_revision = "cbc13b556ace"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlAssessments.__tablename__,
        sa.Column("assessment_id", sa.String(length=50), nullable=False),
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=250), nullable=False),
        sa.Column("assessment_type", sa.String(length=20), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.String(length=250), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("span_id", sa.String(length=50), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("overrides", sa.String(length=50), nullable=True),
        sa.Column("valid", sa.Boolean(), nullable=False, default=True),
        sa.Column("assessment_metadata", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_assessments_trace_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("assessment_id", name="assessments_pk"),
    )

    with op.batch_alter_table(SqlAssessments.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlAssessments.__tablename__}_trace_id_created_timestamp",
            ["trace_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlAssessments.__tablename__}_run_id_created_timestamp",
            ["run_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlAssessments.__tablename__}_last_updated_timestamp",
            ["last_updated_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlAssessments.__tablename__}_assessment_type",
            ["assessment_type"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlAssessments.__tablename__)
