"""add spans table

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


def _get_postgres_version():
    """Get PostgreSQL major version number."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return None

    try:
        from sqlalchemy import text

        result = bind.execute(text("SELECT version()"))
        version_str = result.scalar()
        import re

        match = re.search(r"PostgreSQL (\d+)", version_str)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def upgrade():
    bind = op.get_bind()

    use_generated_column = True

    if bind.dialect.name == "postgresql":
        pg_version = _get_postgres_version()
        if pg_version and pg_version < 12:
            use_generated_column = False

    columns = [
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("parent_span_id", sa.String(length=50), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        # NB: MSSQL doesn't allow TEXT columns in indexes. Limited to 500 chars
        # to stay within MySQL's max index key length of 3072 bytes.
        sa.Column("type", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("start_time_unix_nano", sa.BigInteger(), nullable=False),
        sa.Column("end_time_unix_nano", sa.BigInteger(), nullable=True),
    ]

    if use_generated_column:
        columns.append(
            sa.Column(
                "duration_ns",
                sa.BigInteger(),
                sa.Computed("end_time_unix_nano - start_time_unix_nano", persisted=True),
                nullable=True,
            )
        )
    else:
        columns.append(sa.Column("duration_ns", sa.BigInteger(), nullable=True))

    # NB: LONGTEXT required for MySQL to support large span content (>64KB)
    columns.append(sa.Column("content", sa.Text().with_variant(LONGTEXT, "mysql"), nullable=False))

    op.create_table(
        SqlSpan.__tablename__,
        *columns,
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

    if bind.dialect.name == "postgresql" and not use_generated_column:
        from sqlalchemy import text

        op.execute(
            text(
                """
            CREATE OR REPLACE FUNCTION update_span_duration()
            RETURNS TRIGGER AS $$
            BEGIN
                IF NEW.end_time_unix_nano IS NOT NULL THEN
                    NEW.duration_ns = NEW.end_time_unix_nano - NEW.start_time_unix_nano;
                ELSE
                    NEW.duration_ns = NULL;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
            )
        )

        op.execute(
            text(
                """
            CREATE TRIGGER span_duration_trigger
            BEFORE INSERT OR UPDATE OF end_time_unix_nano, start_time_unix_nano ON spans
            FOR EACH ROW
            EXECUTE FUNCTION update_span_duration();
            """
            )
        )

    with op.batch_alter_table(SqlSpan.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_status_type",
            ["experiment_id", "status", "type"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_type_status",
            ["experiment_id", "type", "status"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_duration",
            ["experiment_id", "duration_ns"],
            unique=False,
        )


def downgrade():
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        from sqlalchemy import text

        op.execute(text("DROP TRIGGER IF EXISTS span_duration_trigger ON spans CASCADE"))
        op.execute(text("DROP FUNCTION IF EXISTS update_span_duration() CASCADE"))

    op.drop_table(SqlSpan.__tablename__)
