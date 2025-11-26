"""add cascade deletion to logged model and span foreign keys

Create Date: 2025-11-26 18:36:56.930151

"""

from alembic import op

from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlSpan,
    SqlTraceInfo,
)

# revision identifiers, used by Alembic.
revision = "c3a777925173"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    """
    Add ON DELETE CASCADE to foreign key constraints from logged_model_* tables,
    trace_info, and spans to experiments table. This ensures child records are
    automatically deleted when an experiment is deleted.
    """
    import sqlalchemy as sa

    # Tables and their foreign key constraint names (excluding spans for now)
    tables_and_constraints = [
        (
            SqlLoggedModelMetric.__tablename__,
            "fk_logged_model_metrics_experiment_id",
        ),
        (
            SqlLoggedModelParam.__tablename__,
            "fk_logged_model_params_experiment_id",
        ),
        (
            SqlLoggedModelTag.__tablename__,
            "fk_logged_model_tags_experiment_id",
        ),
        (
            SqlTraceInfo.__tablename__,
            "fk_trace_info_experiment_id",
        ),
    ]

    # Update all tables (except spans) to add CASCADE deletion
    for table_name, fk_constraint_name in tables_and_constraints:
        # Use batch_alter_table for SQLite compatibility
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            batch_op.drop_constraint(fk_constraint_name, type_="foreignkey")
            batch_op.create_foreign_key(
                fk_constraint_name,
                SqlExperiment.__tablename__,
                ["experiment_id"],
                ["experiment_id"],
                ondelete="CASCADE",
            )

    # Handle spans table separately because it has a computed column (duration_ns).
    # SQLite's batch_alter_table recreates the table by copying data (see
    # alembic.ddl.sqlite.SQLiteImpl.requires_recreate_in_batch), which fails for
    # computed columns (SQLite error: "cannot INSERT into generated column").
    # MySQL/PostgreSQL use native ALTER TABLE (see alembic.ddl.impl.DefaultImpl.
    # requires_recreate_in_batch returns False) and don't need this workaround.
    dialect_name = op.get_context().dialect.name
    if dialect_name == "sqlite":
        # SQLite requires table recreation to modify FK constraints.
        # We must handle the computed column (duration_ns) manually.
        conn = op.get_bind()

        # 1. Create new table with updated FK constraint (ON DELETE CASCADE)
        # fmt: off
        conn.execute(sa.text(
            "CREATE TABLE _alembic_tmp_spans ("
            "trace_id VARCHAR(50) NOT NULL, "
            "experiment_id INTEGER NOT NULL, "
            "span_id VARCHAR(50) NOT NULL, "
            "parent_span_id VARCHAR(50), "
            "name TEXT, "
            "type VARCHAR(500), "
            "status VARCHAR(50) NOT NULL, "
            "start_time_unix_nano BIGINT NOT NULL, "
            "end_time_unix_nano BIGINT, "
            "duration_ns BIGINT GENERATED ALWAYS AS "
            "(end_time_unix_nano - start_time_unix_nano) STORED, "
            "content TEXT NOT NULL, "
            "CONSTRAINT spans_pk PRIMARY KEY (trace_id, span_id), "
            "CONSTRAINT fk_spans_trace_id FOREIGN KEY(trace_id) "
            "REFERENCES trace_info (request_id) ON DELETE CASCADE, "
            "CONSTRAINT fk_spans_experiment_id FOREIGN KEY(experiment_id) "
            "REFERENCES experiments (experiment_id) ON DELETE CASCADE)"
        ))
        # fmt: on

        # 2. Copy data (excluding computed column - it will be auto-calculated)
        conn.execute(
            sa.text("""
            INSERT INTO _alembic_tmp_spans (
                trace_id, experiment_id, span_id, parent_span_id, name, type,
                status, start_time_unix_nano, end_time_unix_nano, content
            )
            SELECT
                trace_id, experiment_id, span_id, parent_span_id, name, type,
                status, start_time_unix_nano, end_time_unix_nano, content
            FROM spans
            """)
        )

        # 3. Drop old table and rename new one
        conn.execute(sa.text("DROP TABLE spans"))
        conn.execute(sa.text("ALTER TABLE _alembic_tmp_spans RENAME TO spans"))

        # 4. Recreate indexes
        conn.execute(sa.text("CREATE INDEX index_spans_experiment_id ON spans (experiment_id)"))
        conn.execute(
            sa.text(
                "CREATE INDEX index_spans_experiment_id_status_type "
                "ON spans (experiment_id, status, type)"
            )
        )
        conn.execute(
            sa.text(
                "CREATE INDEX index_spans_experiment_id_type_status "
                "ON spans (experiment_id, type, status)"
            )
        )
        conn.execute(
            sa.text(
                "CREATE INDEX index_spans_experiment_id_duration "
                "ON spans (experiment_id, duration_ns)"
            )
        )
    else:
        # MySQL/PostgreSQL use native ALTER TABLE - no table recreation, no computed column issue
        with op.batch_alter_table(SqlSpan.__tablename__, schema=None) as batch_op:
            batch_op.drop_constraint("fk_spans_experiment_id", type_="foreignkey")
            batch_op.create_foreign_key(
                "fk_spans_experiment_id",
                SqlExperiment.__tablename__,
                ["experiment_id"],
                ["experiment_id"],
                ondelete="CASCADE",
            )


def downgrade():
    pass
