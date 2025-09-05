"""Fix spans duration column for PostgreSQL 11 compatibility

Revision ID: e2c54d7e0d3f
Revises: de4033877273
Create Date: 2024-01-10 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e2c54d7e0d3f"
down_revision = "71994744cf8e"
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


def _column_exists(table_name, column_name):
    """Check if a column exists in the table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    try:
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception:
        return False


def _table_exists(table_name):
    """Check if a table exists."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade():
    """
    Fix for PostgreSQL 11 users who failed to create spans table due to
    GENERATED column syntax not being supported.

    This migration:
    1. Detects if running on PostgreSQL < 12
    2. Checks if spans table exists but duration_ns is missing (migration failed)
    3. Adds the column as regular column with trigger if needed
    4. Updates existing rows to populate duration_ns
    """
    bind = op.get_bind()

    if bind.dialect.name != "postgresql":
        return

    pg_version = _get_postgres_version()

    if pg_version and pg_version >= 12:
        return

    if not _table_exists("spans"):
        return

    if _column_exists("spans", "duration_ns"):
        return

    # NB: PostgreSQL < 12 with spans table but missing duration_ns column
    from sqlalchemy import text

    with op.batch_alter_table("spans", schema=None) as batch_op:
        batch_op.add_column(sa.Column("duration_ns", sa.BigInteger(), nullable=True))

        try:
            batch_op.create_index(
                "index_spans_experiment_id_duration",
                ["experiment_id", "duration_ns"],
                unique=False,
            )
        except Exception:
            pass

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
        DROP TRIGGER IF EXISTS span_duration_trigger ON spans;
        CREATE TRIGGER span_duration_trigger
        BEFORE INSERT OR UPDATE OF end_time_unix_nano, start_time_unix_nano ON spans
        FOR EACH ROW
        EXECUTE FUNCTION update_span_duration();
        """
        )
    )

    op.execute(
        text(
            """
        UPDATE spans
        SET duration_ns = end_time_unix_nano - start_time_unix_nano
        WHERE end_time_unix_nano IS NOT NULL AND duration_ns IS NULL;
        """
        )
    )


def downgrade():
    """
    Downgrade only affects PostgreSQL < 12 where we added the fix.
    """
    bind = op.get_bind()

    if bind.dialect.name != "postgresql":
        return

    pg_version = _get_postgres_version()
    if pg_version and pg_version >= 12:
        return

    from sqlalchemy import text

    op.execute(text("DROP TRIGGER IF EXISTS span_duration_trigger ON spans CASCADE"))
    op.execute(text("DROP FUNCTION IF EXISTS update_span_duration() CASCADE"))

    # NB: We don't remove the duration_ns column as it may contain data
