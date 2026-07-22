import json
import os
from contextlib import contextmanager
from datetime import date

import pytest
import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.db.workspace_migration import migrate_to_default_workspace
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.tracing.constant import (
    CostKey,
    SpanAttributeKey,
    TokenUsageKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

REVISION = "75868b020152"
PREVIOUS_REVISION = "a8b9c0d1e2f3"
DB_URI = os.environ.get("MLFLOW_TRACKING_URI")
USE_EXTERNAL_DB = DB_URI is not None and not DB_URI.startswith("sqlite")


@pytest.fixture(scope="session", autouse=True)
def _upgrade_external_db_to_head_after_suite():
    yield
    if USE_EXTERNAL_DB:
        command.upgrade(_get_alembic_config(DB_URI), "head")


@contextmanager
def _identity_insert(conn, table_name):
    if conn.dialect.name != "mssql":
        yield
        return

    conn.execute(sa.text(f"SET IDENTITY_INSERT {table_name} ON"))
    try:
        yield
    finally:
        conn.execute(sa.text(f"SET IDENTITY_INSERT {table_name} OFF"))


def _prepare_database(tmp_path):
    if USE_EXTERNAL_DB:
        engine = sa.create_engine(DB_URI)
        with engine.begin() as conn:
            metadata = sa.MetaData()
            metadata.reflect(bind=conn)
            metadata.drop_all(bind=conn)
            InitialBase.metadata.create_all(conn)
        config = _get_alembic_config(DB_URI)
    else:
        db_url = f"sqlite:///{tmp_path / 'trace_analytics_migration.sqlite'}"
        engine = sa.create_engine(db_url)
        InitialBase.metadata.create_all(engine)
        config = _get_alembic_config(db_url)
    command.upgrade(config, PREVIOUS_REVISION)
    return engine, config


def _table(conn, name):
    return sa.Table(name, sa.MetaData(), autoload_with=conn)


def _seed_legacy_analytics_data(conn):
    experiments = _table(conn, "experiments")
    trace_info = _table(conn, "trace_info")
    trace_tags = _table(conn, "trace_tags")
    trace_metadata = _table(conn, "trace_request_metadata")
    trace_metrics = _table(conn, "trace_metrics")
    spans = _table(conn, "spans")
    span_metrics = _table(conn, "span_metrics")
    assessments = _table(conn, "assessments")

    with _identity_insert(conn, "experiments"):
        conn.execute(
            experiments.insert(),
            [
                {
                    "experiment_id": 1,
                    "name": "analytics-experiment",
                    "artifact_location": "file:///tmp/artifacts",
                    "lifecycle_stage": "active",
                    "creation_time": 1,
                    "last_update_time": 1,
                    "workspace": "team-a",
                }
            ],
        )
    conn.execute(
        trace_info.insert(),
        [
            {
                "request_id": "trace-explicit",
                "experiment_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "execution_time_ms": 10,
                "status": "OK",
            },
            {
                "request_id": "trace-fallback",
                "experiment_id": 1,
                "timestamp_ms": 1_700_000_001_000,
                "execution_time_ms": 20,
                "status": "ERROR",
            },
        ],
    )
    conn.execute(
        trace_tags.insert(),
        [
            {
                "request_id": "trace-explicit",
                "key": TraceTagKey.TRACE_NAME,
                "value": "checkout-agent",
            }
        ],
    )
    conn.execute(
        trace_metadata.insert(),
        [
            {
                "request_id": "trace-explicit",
                "key": TraceMetadataKey.TRACE_SESSION,
                "value": "session-123",
            },
            {
                "request_id": "trace-explicit",
                "key": TraceMetadataKey.TOKEN_USAGE,
                "value": json.dumps({
                    TokenUsageKey.INPUT_TOKENS: 11,
                    TokenUsageKey.OUTPUT_TOKENS: 7,
                    TokenUsageKey.TOTAL_TOKENS: 18,
                }),
            },
            {
                "request_id": "trace-explicit",
                "key": TraceMetadataKey.COST,
                "value": json.dumps({
                    CostKey.INPUT_COST: 1.25,
                    CostKey.OUTPUT_COST: 2.5,
                    CostKey.TOTAL_COST: 3.75,
                }),
            },
        ],
    )
    conn.execute(
        trace_metrics.insert(),
        [
            {
                "request_id": "trace-explicit",
                "key": TokenUsageKey.INPUT_TOKENS,
                "value": 12,
            },
            {
                "request_id": "trace-explicit",
                "key": TokenUsageKey.CACHE_READ_INPUT_TOKENS,
                "value": 3,
            },
        ],
    )
    conn.execute(
        spans.insert(),
        [
            {
                "trace_id": "trace-explicit",
                "experiment_id": 1,
                "span_id": "span-explicit",
                "name": "call model",
                "type": "CHAT_MODEL",
                "status": "OK",
                "start_time_unix_nano": 100,
                "end_time_unix_nano": 200,
                "content": "{}",
                "dimension_attributes": json.dumps({
                    SpanAttributeKey.MODEL: "gpt-test",
                    SpanAttributeKey.MODEL_PROVIDER: "test-provider",
                }),
            },
            {
                "trace_id": "trace-fallback",
                "experiment_id": 1,
                "span_id": "span-fallback-1",
                "name": "first call",
                "type": "CHAT_MODEL",
                "status": "OK",
                "start_time_unix_nano": 300,
                "end_time_unix_nano": 500,
                "content": "{}",
                "dimension_attributes": None,
            },
            {
                "trace_id": "trace-fallback",
                "experiment_id": 1,
                "span_id": "span-fallback-2",
                "name": "second call",
                "type": "CHAT_MODEL",
                "status": "OK",
                "start_time_unix_nano": 600,
                "end_time_unix_nano": 900,
                "content": "{}",
                "dimension_attributes": None,
            },
        ],
    )
    conn.execute(
        span_metrics.insert(),
        [
            {
                "trace_id": "trace-explicit",
                "span_id": "span-explicit",
                "key": CostKey.INPUT_COST,
                "value": 99,
            },
            {
                "trace_id": "trace-fallback",
                "span_id": "span-fallback-1",
                "key": CostKey.INPUT_COST,
                "value": 0.5,
            },
            {
                "trace_id": "trace-fallback",
                "span_id": "span-fallback-1",
                "key": CostKey.TOTAL_COST,
                "value": 1.5,
            },
            {
                "trace_id": "trace-fallback",
                "span_id": "span-fallback-2",
                "key": CostKey.INPUT_COST,
                "value": 1.0,
            },
            {
                "trace_id": "trace-fallback",
                "span_id": "span-fallback-2",
                "key": CostKey.TOTAL_COST,
                "value": 2.0,
            },
        ],
    )
    conn.execute(
        assessments.insert(),
        [
            {
                "assessment_id": "assessment-numeric",
                "trace_id": "trace-explicit",
                "name": "quality",
                "assessment_type": "feedback",
                "value": "4.5",
                "created_timestamp": 10,
                "last_updated_timestamp": 10,
                "source_type": "CODE",
                "valid": True,
            },
            {
                "assessment_id": "assessment-yes",
                "trace_id": "trace-explicit",
                "name": "is_safe",
                "assessment_type": "feedback",
                "value": json.dumps("yes"),
                "created_timestamp": 11,
                "last_updated_timestamp": 11,
                "source_type": "HUMAN",
                "valid": True,
            },
            {
                "assessment_id": "assessment-bool",
                "trace_id": "trace-fallback",
                "name": "is_relevant",
                "assessment_type": "expectation",
                "value": "false",
                "created_timestamp": 12,
                "last_updated_timestamp": 12,
                "source_type": "HUMAN",
                "valid": True,
            },
            {
                "assessment_id": "assessment-true",
                "trace_id": "trace-fallback",
                "name": "is_relevant",
                "assessment_type": "expectation",
                "value": "true",
                "created_timestamp": 13,
                "last_updated_timestamp": 13,
                "source_type": "HUMAN",
                "valid": True,
            },
            {
                "assessment_id": "assessment-no",
                "trace_id": "trace-explicit",
                "name": "is_safe",
                "assessment_type": "feedback",
                "value": json.dumps(" No "),
                "created_timestamp": 14,
                "last_updated_timestamp": 14,
                "source_type": "HUMAN",
                "valid": True,
            },
            {
                "assessment_id": "assessment-numeric-string",
                "trace_id": "trace-explicit",
                "name": "quality",
                "assessment_type": "feedback",
                "value": json.dumps("0.8"),
                "created_timestamp": 15,
                "last_updated_timestamp": 15,
                "source_type": "CODE",
                "valid": True,
            },
        ],
    )


def _index_columns(inspector, table_name):
    return {index["name"]: index["column_names"] for index in inspector.get_indexes(table_name)}


def test_trace_analytics_migration_backfills_schema_and_preserves_legacy_rows(tmp_path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)

        command.upgrade(config, REVISION)
        inspector = sa.inspect(engine)

        assert {
            "sql_trace_metric_daily_rollups",
            "sql_span_cost_daily_rollups",
            "sql_assessment_daily_rollups",
            "sql_trace_rollup_rebuild_queue",
        }.issubset(inspector.get_table_names())
        for table_name in (
            "sql_trace_metric_daily_rollups",
            "sql_span_cost_daily_rollups",
            "sql_assessment_daily_rollups",
            "sql_trace_rollup_rebuild_queue",
        ):
            workspace_column = next(
                column
                for column in inspector.get_columns(table_name)
                if column["name"] == "workspace"
            )
            workspace_type = workspace_column["type"]
            assert isinstance(workspace_type, sa.String)
            assert workspace_type.length == 63
            assert workspace_column["nullable"] is False

        span_indexes = _index_columns(inspector, "spans")
        assert span_indexes["idx_spans_cost_trace_time_cover"] == [
            "trace_id",
            "start_time_unix_nano",
        ]
        assert span_indexes["idx_spans_cost_exp_time_cover"] == [
            "experiment_id",
            "start_time_unix_nano",
        ]

        assessment_indexes = _index_columns(inspector, "assessments")
        assert assessment_indexes["idx_assessments_exp_trace_ts"] == [
            "experiment_id",
            "trace_timestamp_ms",
        ]
        assert assessment_indexes["idx_assessments_exp_trace_ts_name"] == [
            "experiment_id",
            "trace_timestamp_ms",
            "name",
        ]
        assert assessment_indexes["idx_assessments_exp_name_valid"] == [
            "experiment_id",
            "name",
            "valid",
        ]

        rollup_indexes = {
            "sql_trace_metric_daily_rollups": (
                "idx_trace_rollups_lookup",
                [
                    "workspace",
                    "experiment_id",
                    "rollup_day",
                    "metric_name",
                    "grouping_set",
                    "trace_status",
                ],
            ),
            "sql_span_cost_daily_rollups": (
                "idx_span_cost_rollups_lookup",
                [
                    "workspace",
                    "experiment_id",
                    "rollup_day",
                    "metric_name",
                    "grouping_set",
                    "model_name",
                    "model_provider",
                ],
            ),
            "sql_assessment_daily_rollups": (
                "idx_assessment_rollups_lookup",
                ["workspace", "experiment_id", "rollup_day", "metric_name", "grouping_set"],
            ),
        }
        for table_name, (index_name, columns) in rollup_indexes.items():
            assert _index_columns(inspector, table_name)[index_name] == columns

        with engine.connect() as conn:
            traces = conn.execute(
                sa.text(
                    "SELECT request_id, trace_name, session_id, input_tokens, output_tokens, "
                    "total_tokens, cache_read_input_tokens, input_cost, output_cost, total_cost "
                    "FROM trace_info ORDER BY request_id"
                )
            ).fetchall()
            assert traces == [
                (
                    "trace-explicit",
                    "checkout-agent",
                    "session-123",
                    12.0,
                    7.0,
                    18.0,
                    3.0,
                    1.25,
                    2.5,
                    3.75,
                ),
                ("trace-fallback", None, None, None, None, None, None, None, None, None),
            ]

            spans = conn.execute(
                sa.text(
                    "SELECT span_id, input_cost, output_cost, total_cost, model_name, "
                    "model_provider FROM spans ORDER BY span_id"
                )
            ).fetchall()
            assert spans == [
                ("span-explicit", 99.0, None, None, "gpt-test", "test-provider"),
                ("span-fallback-1", 0.5, None, 1.5, None, None),
                ("span-fallback-2", 1.0, None, 2.0, None, None),
            ]

            assessments = conn.execute(
                sa.text(
                    "SELECT assessment_id, experiment_id, trace_timestamp_ms, aggregate_value, "
                    "is_numeric_value FROM assessments ORDER BY assessment_id"
                )
            ).fetchall()
            assert assessments == [
                ("assessment-bool", 1, 1_700_000_001_000, 0.0, 0),
                ("assessment-no", 1, 1_700_000_000_000, 0.0, 0),
                ("assessment-numeric", 1, 1_700_000_000_000, 4.5, 1),
                ("assessment-numeric-string", 1, 1_700_000_000_000, None, 0),
                ("assessment-true", 1, 1_700_000_001_000, 1.0, 0),
                ("assessment-yes", 1, 1_700_000_000_000, 1.0, 0),
            ]

            assert conn.execute(sa.text("SELECT COUNT(*) FROM trace_tags")).scalar_one() == 1
            assert (
                conn.execute(sa.text("SELECT COUNT(*) FROM trace_request_metadata")).scalar_one()
                == 3
            )
            assert conn.execute(sa.text("SELECT COUNT(*) FROM trace_metrics")).scalar_one() == 2
            assert conn.execute(sa.text("SELECT COUNT(*) FROM span_metrics")).scalar_one() == 5

        with engine.begin() as conn:
            common_rollup_values = {
                "workspace": "team-a",
                "experiment_id": 1,
                "rollup_day": date(2026, 7, 22),
                "metric_name": "total_cost",
                "grouping_set": "global",
                "sample_count": 2,
            }
            for table_name in (
                "sql_trace_metric_daily_rollups",
                "sql_span_cost_daily_rollups",
                "sql_assessment_daily_rollups",
            ):
                conn.execute(_table(conn, table_name).insert().values(**common_rollup_values))
            conn.execute(
                _table(conn, "sql_trace_rollup_rebuild_queue")
                .insert()
                .values(
                    workspace="team-a",
                    experiment_id=1,
                    rollup_day=date(2026, 7, 22),
                    rollup_family="trace_metrics",
                )
            )

            defaulted_rollup_values = {
                key: value for key, value in common_rollup_values.items() if key != "workspace"
            }
            defaulted_rollup_values["experiment_id"] = 2
            for table_name in (
                "sql_trace_metric_daily_rollups",
                "sql_span_cost_daily_rollups",
                "sql_assessment_daily_rollups",
            ):
                table = _table(conn, table_name)
                conn.execute(table.insert().values(**defaulted_rollup_values))
                assert (
                    conn.execute(
                        sa.select(table.c.workspace).where(table.c.experiment_id == 2)
                    ).scalar_one()
                    == DEFAULT_WORKSPACE_NAME
                )
                conn.execute(table.delete().where(table.c.experiment_id == 2))

            rebuild_queue = _table(conn, "sql_trace_rollup_rebuild_queue")
            conn.execute(
                rebuild_queue.insert().values(
                    experiment_id=2,
                    rollup_day=date(2026, 7, 22),
                    rollup_family="trace_metrics",
                )
            )
            assert (
                conn.execute(
                    sa.select(rebuild_queue.c.workspace).where(rebuild_queue.c.experiment_id == 2)
                ).scalar_one()
                == DEFAULT_WORKSPACE_NAME
            )
            conn.execute(rebuild_queue.delete().where(rebuild_queue.c.experiment_id == 2))

        moved = migrate_to_default_workspace(engine)
        for table_name in (
            "sql_trace_metric_daily_rollups",
            "sql_span_cost_daily_rollups",
            "sql_assessment_daily_rollups",
            "sql_trace_rollup_rebuild_queue",
        ):
            assert moved[table_name] == 1
            with engine.connect() as conn:
                assert (
                    conn.execute(sa.select(_table(conn, table_name).c.workspace)).scalar_one()
                    == "default"
                )
    finally:
        engine.dispose()


def test_trace_analytics_migration_downgrade_and_reupgrade(tmp_path):
    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            _seed_legacy_analytics_data(conn)

        command.upgrade(config, REVISION)
        command.downgrade(config, PREVIOUS_REVISION)

        inspector = sa.inspect(engine)
        assert "trace_name" not in {
            column["name"] for column in inspector.get_columns("trace_info")
        }
        assert "input_cost" not in {column["name"] for column in inspector.get_columns("spans")}
        assert "aggregate_value" not in {
            column["name"] for column in inspector.get_columns("assessments")
        }
        assert "sql_trace_metric_daily_rollups" not in inspector.get_table_names()
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT COUNT(*) FROM trace_tags")).scalar_one() == 1
            assert (
                conn.execute(sa.text("SELECT COUNT(*) FROM trace_request_metadata")).scalar_one()
                == 3
            )
            assert conn.execute(sa.text("SELECT COUNT(*) FROM span_metrics")).scalar_one() == 5
            assert conn.execute(
                sa.text("SELECT duration_ns FROM spans ORDER BY span_id")
            ).scalars().all() == [100, 200, 300]

        command.upgrade(config, REVISION)
        with engine.connect() as conn:
            assert (
                conn.execute(
                    sa.text("SELECT total_cost FROM trace_info WHERE request_id = 'trace-fallback'")
                ).scalar_one()
                is None
            )
    finally:
        engine.dispose()


def test_trace_analytics_migration_rejects_orphaned_assessments(tmp_path):
    if USE_EXTERNAL_DB:
        pytest.skip("DDL rollback behavior is backend-specific")

    engine, config = _prepare_database(tmp_path)
    try:
        with engine.begin() as conn:
            assessments = _table(conn, "assessments")
            conn.execute(
                assessments.insert().values(
                    assessment_id="orphaned-assessment",
                    trace_id="missing-trace",
                    name="quality",
                    assessment_type="feedback",
                    value="1",
                    created_timestamp=10,
                    last_updated_timestamp=10,
                    source_type="CODE",
                    valid=True,
                )
            )

        with pytest.raises(
            RuntimeError,
            match="1 assessments rows have no trace_info row",
        ):
            command.upgrade(config, REVISION)

        inspector = sa.inspect(engine)
        assert "experiment_id" not in {
            column["name"] for column in inspector.get_columns("assessments")
        }
        assert "sql_trace_metric_daily_rollups" not in inspector.get_table_names()
    finally:
        engine.dispose()
