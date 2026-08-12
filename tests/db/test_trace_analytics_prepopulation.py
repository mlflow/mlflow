import json
from decimal import Decimal
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from click.testing import CliRunner
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite

import mlflow.db
import mlflow.store.db.trace_analytics_prepopulation as prepopulation
import mlflow.store.db.utils
from mlflow.store.db import trace_analytics
from mlflow.store.db.utils import _get_alembic_config
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    CostKey,
    SpanAttributeKey,
    TokenUsageKey,
    TraceMetadataKey,
)

from tests.db.test_trace_analytics_migration import (
    DB_URI,
    MIGRATION_MODULE,
    PREVIOUS_REVISION,
    REVISION,
    USE_EXTERNAL_DB,
    _prepare_database,
    _seed_legacy_analytics_data,
    _table,
)


@pytest.fixture(scope="module", autouse=True)
def _upgrade_external_db_to_head_after_module():
    yield
    if USE_EXTERNAL_DB:
        command.upgrade(_get_alembic_config(DB_URI), "head")


def _schema_revision(engine):
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def _upgrade_and_count_analytics_updates(engine, config):
    update_counts = {"trace": 0, "span": 0, "assessment": 0}

    def count_updates(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lstrip().lower()
        entity = next(
            (
                entity
                for table_name, entity in (
                    ("trace_info", "trace"),
                    ("spans", "span"),
                    ("assessments", "assessment"),
                )
                if normalized.startswith(f"update {table_name}")
            ),
            None,
        )
        if entity is not None:
            update_counts[entity] += len(parameters) if executemany else 1

    sa.event.listen(engine, "before_cursor_execute", count_updates)
    try:
        with engine.begin() as conn:
            config.attributes["connection"] = conn
            command.upgrade(config, REVISION)
    finally:
        config.attributes.pop("connection", None)
        sa.event.remove(engine, "before_cursor_execute", count_updates)
    return update_counts


def test_prepopulation_is_non_destructive_and_idempotent(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)

        first = prepopulation.prepopulate_trace_analytics(engine, batch_size=1)
        assert first == prepopulation.TraceAnalyticsPrepopulationStats(
            traces=prepopulation.EntityPrepopulationStats(scanned=2, updated=1),
            spans=prepopulation.EntityPrepopulationStats(scanned=3, updated=3),
            assessments=prepopulation.EntityPrepopulationStats(scanned=6, updated=6),
        )
        assert _schema_revision(engine) == PREVIOUS_REVISION

        inspector = sa.inspect(engine)
        assert "dimension_attributes" in {
            column["name"] for column in inspector.get_columns("spans")
        }
        assert not {
            "sql_trace_metric_daily_rollups",
            "sql_span_cost_daily_rollups",
            "sql_assessment_daily_rollups",
            "sql_trace_rollup_rebuild_queue",
        }.intersection(inspector.get_table_names())
        assert not {
            "idx_spans_cost_trace_time_cover",
            "idx_spans_cost_exp_time_cover",
        }.intersection(index["name"] for index in inspector.get_indexes("spans"))

        with engine.connect() as conn:
            trace_info = _table(conn, "trace_info")
            trace = conn.execute(
                sa.select(
                    trace_info.c.trace_name,
                    trace_info.c.session_id,
                    trace_info.c.input_tokens,
                    trace_info.c.total_cost,
                ).where(trace_info.c.request_id == "trace-explicit")
            ).one()
            assert trace.trace_name is not None
            assert trace.session_id is not None
            assert trace.input_tokens == 12
            assert trace.total_cost == 3.75

            trace_metadata = _table(conn, "trace_request_metadata")
            assert (
                conn.execute(
                    sa.select(sa.func.count()).where(
                        trace_metadata.c.key.in_([
                            TraceMetadataKey.TRACE_SESSION,
                            TraceMetadataKey.TOKEN_USAGE,
                            TraceMetadataKey.COST,
                        ])
                    )
                ).scalar_one()
                == 5
            )
            spans = _table(conn, "spans")
            assert (
                conn.execute(
                    sa.select(sa.func.count()).where(spans.c.dimension_attributes.isnot(None))
                ).scalar_one()
                == 3
            )

        second = prepopulation.prepopulate_trace_analytics(engine, batch_size=1)
        assert second.traces.updated == 0
        assert second.spans.updated == 0
        assert second.assessments.updated == 0
    finally:
        engine.dispose()


def test_prepopulation_resumes_after_a_committed_batch(tmp_path, monkeypatch):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
            conn.execute(
                _table(conn, "trace_request_metadata")
                .insert()
                .values(
                    request_id="trace-fallback",
                    key=TraceMetadataKey.TRACE_SESSION,
                    value="fallback-session",
                )
            )

        original_trace_batch = prepopulation._trace_batch
        call_count = 0

        def fail_on_second_trace_batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated interruption")
            return original_trace_batch(*args, **kwargs)

        monkeypatch.setattr(prepopulation, "_trace_batch", fail_on_second_trace_batch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            prepopulation.prepopulate_trace_analytics(engine, batch_size=1)

        with engine.connect() as conn:
            trace_info = _table(conn, "trace_info")
            traces = {
                row.request_id: row.session_id
                for row in conn.execute(sa.select(trace_info.c.request_id, trace_info.c.session_id))
            }
            assert traces["trace-explicit"] is not None
            assert traces["trace-fallback"] is None

        monkeypatch.setattr(prepopulation, "_trace_batch", original_trace_batch)
        stats = prepopulation.prepopulate_trace_analytics(engine, batch_size=1)
        assert stats.traces.updated == 1
        with engine.connect() as conn:
            trace_info = _table(conn, "trace_info")
            assert (
                conn.execute(
                    sa.select(trace_info.c.session_id).where(
                        trace_info.c.request_id == "trace-fallback"
                    )
                ).scalar_one()
                == "fallback-session"
            )
    finally:
        engine.dispose()


def test_prepopulation_reports_monotonic_progress(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)

        progress = {"traces": [], "spans": [], "assessments": []}
        prepopulation.prepopulate_trace_analytics(
            engine,
            batch_size=1,
            progress_every_batches=1,
            progress_callback=lambda entity, stats: progress[entity].append(stats.scanned),
        )

        assert progress == {
            "traces": [1, 2],
            "spans": [1, 2, 3],
            "assessments": [1, 2, 3, 4, 5, 6],
        }
    finally:
        engine.dispose()


def test_prepopulation_completes_a_partially_expanded_schema(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
            Operations(MigrationContext.configure(conn)).add_column(
                "trace_info",
                sa.Column(
                    "trace_name",
                    sa.String(length=MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE),
                    nullable=True,
                ),
            )

        prepopulation.prepopulate_trace_analytics(engine)

        trace_columns = {column["name"] for column in sa.inspect(engine).get_columns("trace_info")}
        assert set(prepopulation._TRACE_VALUE_COLUMNS).issubset(trace_columns)
        assert _schema_revision(engine) == PREVIOUS_REVISION
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("dialect_name", "previous_timeout", "expected_statements"),
    [
        ("postgresql", None, ["SET LOCAL lock_timeout = '5000ms'"]),
        (
            "mysql",
            31_536_000,
            [
                "SELECT @@SESSION.lock_wait_timeout",
                "SET SESSION lock_wait_timeout = 5",
                "SET SESSION lock_wait_timeout = 31536000",
            ],
        ),
        (
            "mssql",
            -1,
            ["SELECT @@LOCK_TIMEOUT", "SET LOCK_TIMEOUT 5000", "SET LOCK_TIMEOUT -1"],
        ),
        ("sqlite", None, []),
    ],
)
def test_ddl_lock_timeout_is_scoped_by_dialect(dialect_name, previous_timeout, expected_statements):
    connection = Mock()
    connection.dialect.name = dialect_name
    connection.exec_driver_sql.return_value.scalar_one.return_value = previous_timeout

    with trace_analytics._ddl_lock_timeout(connection, timeout_seconds=5):
        pass

    statements = [call.args[0] for call in connection.exec_driver_sql.call_args_list]
    assert statements == expected_statements


def test_prepopulation_reports_ddl_lock_failure(tmp_path, monkeypatch):
    engine, _ = _prepare_database(tmp_path)
    try:
        monkeypatch.setattr(
            prepopulation,
            "ensure_analytics_columns",
            Mock(
                side_effect=sa.exc.OperationalError(
                    "ALTER TABLE trace_info",
                    {},
                    RuntimeError("lock timeout"),
                )
            ),
        )

        with pytest.raises(RuntimeError, match="retry during a low-traffic period"):
            prepopulation.prepopulate_trace_analytics(engine)
    finally:
        engine.dispose()


def test_prepopulation_clears_stale_values_and_preserves_metric_precedence(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
        prepopulation.prepopulate_trace_analytics(engine)

        with engine.begin() as conn:
            trace_metadata = _table(conn, "trace_request_metadata")
            trace_metrics = _table(conn, "trace_metrics")
            conn.execute(
                trace_metadata.delete().where(
                    trace_metadata.c.request_id == "trace-explicit",
                    trace_metadata.c.key == TraceMetadataKey.TRACE_SESSION,
                )
            )
            conn.execute(
                trace_metrics
                .update()
                .where(
                    trace_metrics.c.request_id == "trace-explicit",
                    trace_metrics.c.key == TokenUsageKey.INPUT_TOKENS,
                )
                .values(value=99)
            )

        stats = prepopulation.prepopulate_trace_analytics(engine)
        assert stats.traces.updated == 1
        with engine.connect() as conn:
            trace_info = _table(conn, "trace_info")
            # input_tokens comes from the trace_metric (99), taking precedence over the token_usage
            # metadata value (11); the deleted session metadata clears the stale session_id.
            assert conn.execute(
                sa.select(trace_info.c.session_id, trace_info.c.input_tokens).where(
                    trace_info.c.request_id == "trace-explicit"
                )
            ).one() == (None, 99)
    finally:
        engine.dispose()


def test_prepopulation_rejects_an_incompatible_existing_column(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            Operations(MigrationContext.configure(conn)).add_column(
                "trace_info", sa.Column("trace_name", sa.Integer(), nullable=True)
            )

        with pytest.raises(
            RuntimeError,
            match=r"existing column trace_info\.trace_name has incompatible schema",
        ):
            prepopulation.prepopulate_trace_analytics(engine)
        assert "session_id" not in {
            column["name"] for column in sa.inspect(engine).get_columns("trace_info")
        }
    finally:
        engine.dispose()


def test_prepopulation_rejects_an_incompatible_existing_server_default(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            Operations(MigrationContext.configure(conn)).add_column(
                "assessments",
                sa.Column(
                    "is_numeric_value",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.true(),
                ),
            )

        with pytest.raises(
            RuntimeError,
            match=r"existing column assessments\.is_numeric_value.*server default",
        ):
            prepopulation.prepopulate_trace_analytics(engine)
    finally:
        engine.dispose()


def test_prepopulation_preserves_unpromoted_dimension_content(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
            spans = _table(conn, "spans")
            conn.execute(
                spans
                .update()
                .where(spans.c.span_id == "span-explicit")
                .values(dimension_attributes=json.dumps({"custom.dimension": "value"}))
            )

        prepopulation.prepopulate_trace_analytics(engine)

        with engine.connect() as conn:
            spans = _table(conn, "spans")
            row = conn.execute(
                sa.select(spans.c.dimension_attributes, spans.c.model_name).where(
                    spans.c.span_id == "span-explicit"
                )
            ).one()
            assert json.loads(row.dimension_attributes) == {"custom.dimension": "value"}
            assert row.model_name is None
    finally:
        engine.dispose()


def test_final_migration_repairs_legacy_writes_after_prepopulation(tmp_path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
        prepopulation.prepopulate_trace_analytics(engine)

        with engine.begin() as conn:
            trace_metadata = _table(conn, "trace_request_metadata")
            trace_metrics = _table(conn, "trace_metrics")
            span_metrics = _table(conn, "span_metrics")
            assessments = _table(conn, "assessments")
            conn.execute(
                trace_metadata
                .update()
                .where(
                    trace_metadata.c.request_id == "trace-explicit",
                    trace_metadata.c.key == TraceMetadataKey.TRACE_SESSION,
                )
                .values(value="new-session")
            )
            conn.execute(
                trace_metrics
                .update()
                .where(
                    trace_metrics.c.request_id == "trace-explicit",
                    trace_metrics.c.key == TokenUsageKey.INPUT_TOKENS,
                )
                .values(value=25)
            )
            conn.execute(
                span_metrics
                .update()
                .where(
                    span_metrics.c.trace_id == "trace-explicit",
                    span_metrics.c.span_id == "span-explicit",
                    span_metrics.c.key == CostKey.INPUT_COST,
                )
                .values(value=101.0)
            )
            conn.execute(
                assessments
                .update()
                .where(assessments.c.assessment_id == "assessment-numeric")
                .values(value=json.dumps(8.5))
            )

        update_counts = _upgrade_and_count_analytics_updates(engine, config)
        assert update_counts == {"trace": 1, "span": 1, "assessment": 1}

        with engine.connect() as conn:
            trace_info = _table(conn, "trace_info")
            trace = conn.execute(
                sa.select(
                    trace_info.c.session_id,
                    trace_info.c.input_tokens,
                ).where(trace_info.c.request_id == "trace-explicit")
            ).one()
            assert trace == ("new-session", 25)

            spans = _table(conn, "spans")
            assert (
                conn.execute(
                    sa.select(spans.c.input_cost).where(
                        spans.c.trace_id == "trace-explicit",
                        spans.c.span_id == "span-explicit",
                    )
                ).scalar_one()
                == 101.0
            )
            assessments = _table(conn, "assessments")
            assert (
                conn.execute(
                    sa.select(assessments.c.aggregate_value).where(
                        assessments.c.assessment_id == "assessment-numeric"
                    )
                ).scalar_one()
                == 8.5
            )

        # The final migration promotes model dimensions into columns and then drops the raw
        # dimension_attributes JSON blob, so it must be gone once the upgrade completes.
        assert "dimension_attributes" not in {
            column["name"] for column in sa.inspect(engine).get_columns("spans")
        }
        assert _schema_revision(engine) == REVISION
    finally:
        engine.dispose()


def test_final_migration_skips_fully_prepopulated_rows(tmp_path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
        prepopulation.prepopulate_trace_analytics(engine)

        update_counts = _upgrade_and_count_analytics_updates(engine, config)
        assert update_counts == {"trace": 0, "span": 0, "assessment": 0}
    finally:
        engine.dispose()


def test_prepopulation_rejects_the_wrong_revision(tmp_path):
    engine, config = _prepare_database(tmp_path)
    try:
        command.upgrade(config, REVISION)
        with pytest.raises(RuntimeError, match="must run before the trace analytics migration"):
            prepopulation.prepopulate_trace_analytics(engine)
    finally:
        engine.dispose()


def test_prepopulation_rejects_an_empty_database_without_bootstrapping(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'empty.sqlite'}")
    try:
        with pytest.raises(RuntimeError, match="found no Alembic revision"):
            prepopulation.prepopulate_trace_analytics(engine)
        assert not {
            "trace_info",
            "spans",
            "assessments",
        }.intersection(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_prepopulation_schema_contract_matches_orm_models(tmp_path):
    from mlflow.store.tracking.dbmodels.models import SqlAssessments, SqlSpan, SqlTraceInfo

    engine, _ = _prepare_database(tmp_path)
    try:
        models = {
            "trace_info": SqlTraceInfo,
            "assessments": SqlAssessments,
            "spans": SqlSpan,
        }
        for table_name, expected_columns in trace_analytics.analytics_columns_by_table().items():
            model_table = models[table_name].__table__
            for expected in expected_columns:
                actual = model_table.c[expected.name]
                assert trace_analytics.types_are_compatible(
                    expected.type, actual.type, engine.dialect
                )
                if expected.name == "is_numeric_value":
                    # The prepopulation helper adds is_numeric_value nullable for a fast online
                    # schema change; the offline migration tightens it to NOT NULL to match the ORM.
                    assert expected.nullable
                    assert not actual.nullable
                else:
                    assert expected.nullable == actual.nullable
                assert (expected.server_default is None) == (actual.server_default is None)
                if expected.server_default is not None:
                    assert trace_analytics._normalized_false_default(actual.server_default.arg)
    finally:
        engine.dispose()


def test_frozen_migration_columns_match_the_live_helper():
    live_columns_by_table = trace_analytics.analytics_columns_by_table()
    frozen_columns_by_table = MIGRATION_MODULE._analytics_columns_by_table()
    assert live_columns_by_table.keys() == frozen_columns_by_table.keys()

    dialect = sqlite.dialect()
    for table_name, live_columns in live_columns_by_table.items():
        frozen_columns = {column.name: column for column in frozen_columns_by_table[table_name]}
        assert {column.name for column in live_columns} == set(frozen_columns)
        for live_column in live_columns:
            frozen_column = frozen_columns[live_column.name]
            assert trace_analytics.types_are_compatible(
                live_column.type, frozen_column.type, dialect
            )
            assert trace_analytics.types_are_compatible(
                frozen_column.type, live_column.type, dialect
            )
            assert live_column.nullable == frozen_column.nullable
            assert (live_column.server_default is None) == (frozen_column.server_default is None)


@pytest.mark.parametrize(
    ("expected", "actual", "dialect"),
    [
        # psycopg reflects Float(53) as DOUBLE_PRECISION, and Float.dialect_impl() on the psycopg
        # dialect yields _PsycopgFloat (a Numeric subclass, not sa.Float). Guards that a FLOAT(53)
        # column is not rejected as incompatible with itself on PostgreSQL (SQLite never triggers).
        (sa.Float(precision=53), postgresql.DOUBLE_PRECISION(precision=53), postgresql.dialect()),
        # MySQL maps long String columns to TEXT via with_variant(), so a reflected TEXT column must
        # match an expected String on that dialect.
        (
            sa.String(length=8000).with_variant(sa.Text(), "mysql"),
            mysql.TEXT(),
            mysql.dialect(),
        ),
        # MySQL has no native BOOLEAN type: it stores and reflects Boolean columns as TINYINT(1),
        # so a reflected TINYINT(1) must match an expected Boolean on that dialect (e.g. when a
        # prepopulation rerun revalidates the is_numeric_value column it already added).
        (sa.Boolean(), mysql.TINYINT(display_width=1), mysql.dialect()),
    ],
)
def test_types_are_compatible_across_dialect_reflected_types(expected, actual, dialect):
    assert trace_analytics.types_are_compatible(expected, actual, dialect)


@pytest.mark.parametrize(
    ("expected", "actual", "dialect"),
    [
        # A wider TINYINT (e.g. a real 0-255 integer column) is not a Boolean and must be rejected.
        (sa.Boolean(), mysql.TINYINT(display_width=4), mysql.dialect()),
        (sa.Boolean(), mysql.INTEGER(), mysql.dialect()),
    ],
)
def test_types_are_incompatible_for_non_boolean_tinyint(expected, actual, dialect):
    assert not trace_analytics.types_are_compatible(expected, actual, dialect)


def test_prepopulation_conversion_semantics_match_the_frozen_migration():
    numeric_values = [
        None,
        True,
        False,
        0,
        100,
        100.0,
        Decimal("100"),
        1.5,
        float("nan"),
        float("inf"),
        -(2**63),
        2**63 - 1,
        -(2**63) - 1,
        2**63,
        "not-a-number",
    ]
    for value in numeric_values:
        assert MIGRATION_MODULE._finite_float_or_none(value) == prepopulation.finite_float_or_none(
            value
        )

    for value in numeric_values:
        assert MIGRATION_MODULE._token_count_or_none(value) == prepopulation.token_count_or_none(
            value
        )

    string_values = [
        None,
        123,
        "",
        "abc",
        "x" * (MAX_CHARS_IN_TRACE_INFO_METADATA - 1),
        "x" * MAX_CHARS_IN_TRACE_INFO_METADATA,
        "x" * (MAX_CHARS_IN_TRACE_INFO_METADATA + 1),
        "y" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
        "y" * (MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE + 1),
    ]
    for value in string_values:
        for max_length in (MAX_CHARS_IN_TRACE_INFO_METADATA, MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE):
            migration_value, _ = MIGRATION_MODULE._bounded_string_or_none(value, max_length)
            assert migration_value == prepopulation._bounded_string_or_none(value, max_length)

    assessment_values = [None, True, False, 1, 1.5, " yes ", "NO", "0.8", "text"]
    for value in assessment_values:
        assert MIGRATION_MODULE._assessment_aggregate(
            json.dumps(value)
        ) == prepopulation.assessment_aggregate(value)

    dimension_values = [
        None,
        {},
        {SpanAttributeKey.MODEL: None},
        {SpanAttributeKey.MODEL: "model"},
        {SpanAttributeKey.MODEL_PROVIDER: "provider"},
        {
            SpanAttributeKey.MODEL: "model",
            SpanAttributeKey.MODEL_PROVIDER: "provider",
        },
    ]
    for value in dimension_values:
        migration_dimensions = MIGRATION_MODULE._json_object(value)
        prepopulation_dimensions = prepopulation._json_object(value)
        assert migration_dimensions == prepopulation_dimensions
        for dimension_value in migration_dimensions.values():
            migration_value, _ = MIGRATION_MODULE._bounded_string_or_none(
                dimension_value,
                500,
            )
            assert migration_value == prepopulation.bounded_model_dimension(dimension_value)


@pytest.mark.parametrize(
    ("dialect_name", "dialect", "expected_sql"),
    [
        ("sqlite", sqlite.dialect(), "IN (VALUES"),
        ("postgresql", postgresql.dialect(), " IN (("),
        ("mysql", mysql.dialect(), " IN (("),
        ("mssql", mssql.dialect(), "JOIN (VALUES"),
    ],
)
def test_prepopulation_span_metric_lookup_compiles_for_supported_dialects(
    dialect_name, dialect, expected_sql
):
    span_metrics = sa.table(
        "span_metrics",
        sa.column("trace_id", sa.String(50)),
        sa.column("span_id", sa.String(50)),
        sa.column("key", sa.String(250)),
        sa.column("value", sa.Float()),
    )
    span_keys = [("trace-1", f"span-{index}") for index in range(prepopulation.DEFAULT_BATCH_SIZE)]

    query = prepopulation._span_metrics_for_keys_query(span_metrics, span_keys, dialect_name)
    compiled = query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
    sql = str(compiled)

    assert expected_sql in sql
    assert " OR " not in sql
    assert len(compiled.params) == (
        2 * prepopulation.DEFAULT_BATCH_SIZE + len(prepopulation.COST_COLUMN_BY_KEY)
    )


def test_prepopulation_cli(tmp_path):
    engine, _ = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)
        result = CliRunner().invoke(
            mlflow.db.commands,
            ["prepopulate-trace-analytics", "--batch-size", "1"],
            env={"MLFLOW_TRACKING_URI": engine.url.render_as_string(hide_password=False)},
        )
        assert result.exit_code == 0, result.output
        assert "Traces progress: scanned=1, updated=1" in result.output
        assert "Spans progress: scanned=1, updated=1" in result.output
        assert "Assessments progress: scanned=1, updated=1" in result.output
        assert "Traces: scanned=2, updated=1" in result.output
        assert "Spans: scanned=3, updated=3" in result.output
        assert "Assessments: scanned=6, updated=6" in result.output
        assert "without advancing the Alembic revision" in result.output
        assert "Run `mlflow db upgrade`" in result.output
    finally:
        engine.dispose()


def test_prepopulation_cli_reports_errors_and_disposes_engine(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(
        mlflow.store.db.utils,
        "create_sqlalchemy_engine_with_retry",
        lambda _: engine,
    )
    monkeypatch.setattr(
        prepopulation,
        "prepopulate_trace_analytics",
        Mock(side_effect=RuntimeError("prepopulation failed")),
    )

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["prepopulate-trace-analytics", "sqlite:///unused.db"],
    )

    assert result.exit_code == 1
    assert "Error: prepopulation failed" in result.output
    engine.dispose.assert_called_once_with()


def test_prepopulation_cli_does_not_expose_database_credentials(monkeypatch):
    database_url = "postgresql://trace_user:super-secret@database.example/mlflow"
    monkeypatch.setattr(
        mlflow.store.db.utils,
        "create_sqlalchemy_engine_with_retry",
        Mock(
            side_effect=sa.exc.OperationalError(
                f"connect to {database_url}",
                {},
                RuntimeError(database_url),
            )
        ),
    )

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["prepopulate-trace-analytics"],
        env={"MLFLOW_TRACKING_URI": database_url},
    )

    assert result.exit_code == 1
    assert "Database operation failed (OperationalError)" in result.output
    assert "super-secret" not in result.output


def test_prepopulation_tolerates_rows_deleted_by_the_live_server():
    result = Mock(rowcount=1)
    result.supports_sane_multi_rowcount.return_value = True
    connection = Mock()
    connection.execute.return_value = result

    updated = prepopulation._execute_updates(
        connection,
        "update statement",
        [{"id": "trace-1"}, {"id": "trace-2"}],
    )

    assert updated == 1


@pytest.mark.parametrize("batch_size", ["0", "251"])
def test_prepopulation_cli_rejects_unsafe_batch_sizes(batch_size):
    result = CliRunner().invoke(
        mlflow.db.commands,
        ["prepopulate-trace-analytics", "sqlite:///unused.db", "--batch-size", batch_size],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--batch-size'" in result.output
