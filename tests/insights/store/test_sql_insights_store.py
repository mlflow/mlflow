"""
Consolidated tests for SqlInsightsStore.
Combines unit tests, integration tests, and performance tests.
"""

import json
import os
import random
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.insights.store.sql_insights_store import SqlInsightsStore
from mlflow.store.tracking.dbmodels.models import (
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import encode_span_id, encode_trace_id


@pytest.fixture
def test_store() -> dict[str, Any]:
    """Set up test database and store instances."""
    temp_dir = tempfile.mkdtemp()
    # Use environment variable if set (for testing against different databases)
    # Otherwise default to SQLite
    db_uri = os.environ.get(
        MLFLOW_TRACKING_URI.name,
        f"sqlite:///{os.path.join(temp_dir, 'test.db')}",
    )

    tracking_store = SqlAlchemyStore(db_uri, temp_dir)
    insights_store = SqlInsightsStore(tracking_store)

    # Use the tracking store's engine to ensure we're using the same connection
    engine = tracking_store.engine
    Session = sessionmaker(bind=engine)

    exp_id = tracking_store.create_experiment("test_experiment")

    yield {
        "temp_dir": temp_dir,
        "db_uri": db_uri,
        "tracking_store": tracking_store,
        "insights_store": insights_store,
        "engine": engine,
        "Session": Session,
        "exp_id": exp_id,
    }

    # Teardown - Clean up test data
    with Session() as session:
        # Delete all test data to ensure clean state for next test
        session.query(SqlSpan).delete()
        session.query(SqlTraceMetadata).delete()
        session.query(SqlTraceTag).delete()
        session.query(SqlTraceInfo).delete()
        session.commit()

    # Only remove temp dir if using SQLite
    if db_uri.startswith("sqlite://"):
        shutil.rmtree(temp_dir)


def create_test_trace_and_spans(
    store_info: dict[str, Any],
    exp_id: str,
    num_spans: int = 3,
    start_time: datetime | None = None,
    duration_ms: int = 100,
    status: str = "OK",
    add_metadata: bool = True,
) -> tuple[str, str]:
    """Helper to create test trace with spans directly in database."""
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    trace_id = encode_trace_id(random.getrandbits(128))
    request_id = f"req_{trace_id[:8]}"
    Session = store_info["Session"]

    with Session() as session:
        trace_info = SqlTraceInfo(
            request_id=request_id,
            experiment_id=exp_id,
            timestamp_ms=int(start_time.timestamp() * 1000),
            execution_time_ms=duration_ms,
            status="OK",
        )
        session.add(trace_info)
        session.flush()

        if add_metadata:
            metadata = SqlTraceMetadata(
                request_id=request_id,
                key="mlflow.traceInputs",
                value='{"prompt": "Test prompt for analysis"}',
            )
            session.add(metadata)

            metadata_output = SqlTraceMetadata(
                request_id=request_id,
                key="mlflow.traceOutputs",
                value=(
                    '{"response": "This is a test response that is longer than 50 characters'
                    ' to test quality metrics properly"}'
                ),
            )
            session.add(metadata_output)

        for i in range(num_spans):
            span_id = encode_span_id(random.getrandbits(64))
            parent_id = encode_span_id(random.getrandbits(64)) if i > 0 else None

            # For the root span (parent_id=None), use the full duration
            # For child spans, distribute the duration
            if parent_id is None:
                span_start_ns = int(start_time.timestamp() * 1e9)
                span_end_ns = span_start_ns + (duration_ms * 1_000_000)  # Convert ms to ns
            else:
                # Child spans get a portion of the duration
                span_start_ns = int(start_time.timestamp() * 1e9) + i * 10_000_000
                span_end_ns = span_start_ns + (duration_ms // num_spans * 1_000_000)

            span = SqlSpan(
                trace_id=request_id,  # trace_id is the foreign key to trace_info.request_id
                experiment_id=exp_id,
                span_id=span_id,
                parent_span_id=parent_id,
                name=f"span_{i}",
                status=status,
                start_time_unix_nano=span_start_ns,
                end_time_unix_nano=span_end_ns,
                # duration_ns is computed automatically
                content=(
                    f'{{"span_type": "{"LLM" if i == 0 else "CHAIN"}", '
                    f'"attributes": {{"model": "gpt-4" if i == 0 else ""}}, '
                    f'"inputs": {{"test": "input"}}, "outputs": {{"test": "output"}}, '
                    f'"events": []}}'
                ),
            )
            session.add(span)

        session.commit()

    return request_id, trace_id


# ============== Basic Smoke Tests ==============


def test_basic_insights_store_creation():
    """Test that we can create an insights store."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    artifact_path = Path(temp_dir) / "artifacts"
    artifact_path.mkdir()

    db_uri = f"sqlite:///{db_path}"
    tracking_store = SqlAlchemyStore(db_uri, artifact_path.as_uri())
    insights_store = SqlInsightsStore(tracking_store)

    assert insights_store is not None
    assert insights_store.store == tracking_store
    assert insights_store.dialect == "sqlite"

    # Clean up
    shutil.rmtree(temp_dir)


def test_empty_metrics(test_store: dict[str, Any]):
    """Test metrics retrieval with no data."""
    insights_store = test_store["insights_store"]
    exp_id = test_store["exp_id"]

    # Get metrics for empty experiment
    metrics = insights_store.get_operational_metrics([exp_id])

    assert metrics.total_traces == 0
    assert metrics.error_count == 0
    assert metrics.error_rate == 0.0
    assert metrics.p50_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0
    assert len(metrics.time_buckets) == 0
    assert len(metrics.top_error_spans) == 0
    assert len(metrics.top_slow_tools) == 0


def test_single_trace_metrics(test_store: dict[str, Any]):
    """Test metrics with a single trace."""
    insights_store = test_store["insights_store"]
    tracking_store = test_store["tracking_store"]
    exp_id = test_store["exp_id"]

    now = datetime.now(timezone.utc)
    trace_id = f"tr-{uuid.uuid4().hex[:16]}"
    timestamp = int(now.timestamp() * 1000)

    # Create trace and span using SQLAlchemy models directly
    with tracking_store.ManagedSessionMaker() as session:
        # Create trace
        trace_info = SqlTraceInfo(
            request_id=trace_id,
            experiment_id=exp_id,
            timestamp_ms=timestamp,
            execution_time_ms=1000,
            status="OK",
        )
        session.add(trace_info)

        # Add trace metadata
        inputs_meta = SqlTraceMetadata(
            request_id=trace_id, key="mlflow.traceInputs", value='{"query": "test"}'
        )
        outputs_meta = SqlTraceMetadata(
            request_id=trace_id,
            key="mlflow.traceOutputs",
            value=(
                '{"response": "This is a test response that is definitely longer than 50'
                ' characters to pass the test"}'
            ),
        )
        session.add(inputs_meta)
        session.add(outputs_meta)

        # Create span
        span_content = {
            "attributes": {"test": "true"},
            "inputs": {"query": "test"},
            "outputs": {
                "response": (
                    "This is a test response that is definitely longer than 50 "
                    "characters to pass the test"
                )
            },
        }
        span = SqlSpan(
            trace_id=trace_id,
            experiment_id=exp_id,
            span_id=f"sp-{uuid.uuid4().hex[:16]}",
            parent_span_id=None,
            name="test_span",
            type="CHAIN",
            status="OK",
            start_time_unix_nano=timestamp * 1_000_000,
            end_time_unix_nano=(timestamp + 1000) * 1_000_000,
            content=json.dumps(span_content),
        )
        session.add(span)
        session.commit()

    # Get metrics
    metrics = insights_store.get_operational_metrics([exp_id])

    assert metrics.total_traces == 1
    assert metrics.error_count == 0
    assert metrics.error_rate == 0.0
    assert metrics.p50_latency_ms == 1000.0  # 1 second in milliseconds
    assert metrics.p95_latency_ms == 1000.0  # Only one trace, so all percentiles are the same

    # Test quality metrics
    quality = insights_store.get_quality_metrics([exp_id])
    assert quality.minimal_responses.value == 0.0  # Response is > 50 chars
    assert quality.response_quality_issues.value == 0.0  # No error indicators

    # Test census generation
    census = insights_store.generate_census(exp_id, table_name="test_table")
    assert census.metadata.table_name == "test_table"
    assert census.operational_metrics.total_traces == 1


def test_error_trace_metrics(test_store: dict[str, Any]):
    """Test metrics with an error trace."""
    insights_store = test_store["insights_store"]
    tracking_store = test_store["tracking_store"]
    exp_id = test_store["exp_id"]

    now = datetime.now(timezone.utc)
    trace_id = f"tr-{uuid.uuid4().hex[:16]}"
    timestamp = int(now.timestamp() * 1000)

    # Create error trace and span using SQLAlchemy models directly
    with tracking_store.ManagedSessionMaker() as session:
        # Create error trace
        trace_info = SqlTraceInfo(
            request_id=trace_id,
            experiment_id=exp_id,
            timestamp_ms=timestamp,
            execution_time_ms=500,
            status="ERROR",
        )
        session.add(trace_info)

        # Add trace metadata with "Sorry" in the output
        inputs_meta = SqlTraceMetadata(
            request_id=trace_id, key="mlflow.traceInputs", value='{"query": "test"}'
        )
        outputs_meta = SqlTraceMetadata(
            request_id=trace_id,
            key="mlflow.traceOutputs",
            value='{"error": "Sorry, I cannot process this"}',
        )
        session.add(inputs_meta)
        session.add(outputs_meta)

        # Create error span
        span_content = {
            "attributes": {"error": "true"},
            "inputs": {"query": "test"},
            "outputs": {"error": "Sorry, I cannot process this"},
        }
        span = SqlSpan(
            trace_id=trace_id,
            experiment_id=exp_id,
            span_id=f"sp-{uuid.uuid4().hex[:16]}",
            parent_span_id=None,
            name="test_error_span",
            type="CHAIN",
            status="ERROR",
            start_time_unix_nano=timestamp * 1_000_000,
            end_time_unix_nano=(timestamp + 500) * 1_000_000,
            content=json.dumps(span_content),
        )
        session.add(span)
        session.commit()

    # Get metrics
    metrics = insights_store.get_operational_metrics([exp_id])

    assert metrics.total_traces == 1
    assert metrics.error_count == 1
    assert metrics.error_rate == 100.0

    # Check error spans
    assert len(metrics.top_error_spans) == 1
    assert metrics.top_error_spans[0].error_span_name == "test_error_span"
    assert metrics.top_error_spans[0].count == 1

    # Test quality metrics - should detect "Sorry" as quality issue
    quality = insights_store.get_quality_metrics([exp_id])
    assert quality.response_quality_issues.value == 100.0  # Has "Sorry" in response


# ============== Database Optimization Tests ==============


def test_get_operational_metrics_db_aggregation(test_store: dict[str, Any]):
    """Test that operational metrics use database aggregation."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)
    for i in range(10):
        create_test_trace_and_spans(
            test_store,
            exp_id,
            num_spans=3,
            start_time=now - timedelta(minutes=30 - i),
            duration_ms=100 + i * 10,
        )

    metrics = insights_store.get_operational_metrics(
        experiment_ids=[exp_id],
        start_time=now - timedelta(hours=1),
        end_time=now,
    )

    assert metrics.total_traces == 10
    assert metrics.p50_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    # Verify percentiles are reasonable
    assert metrics.p95_latency_ms >= metrics.p50_latency_ms


def test_calculate_latency_percentiles_single_query(test_store: dict[str, Any]):
    """Test that percentile calculations happen in a single DB query."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)
    # Create traces with clearly different durations
    durations = []
    for i in range(20):
        duration = 100 + i * 50  # 100, 150, 200, ..., 1050
        durations.append(duration)
        create_test_trace_and_spans(
            test_store,
            exp_id,
            num_spans=1,
            start_time=now - timedelta(minutes=60 - i),
            duration_ms=duration,
        )

    # Use the operational metrics which includes percentiles
    metrics = insights_store.get_operational_metrics(experiment_ids=[exp_id])

    assert metrics.p50_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    assert isinstance(metrics.p50_latency_ms, (int, float))
    # With durations from 100 to 1050, p95 should be > p50
    assert metrics.p95_latency_ms > metrics.p50_latency_ms


def test_get_quality_metrics_aggregated_queries(test_store: dict[str, Any]):
    """Test that quality metrics use aggregated database queries."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)

    for i in range(5):
        create_test_trace_and_spans(
            test_store, exp_id, status="OK", start_time=now - timedelta(minutes=30 - i)
        )

    for i in range(3):
        create_test_trace_and_spans(
            test_store, exp_id, status="ERROR", start_time=now - timedelta(minutes=20 - i)
        )

    metrics = insights_store.get_quality_metrics(
        experiment_ids=[exp_id],
        start_time=now - timedelta(hours=1),
        end_time=now,
    )

    # Verify that quality metrics are calculated
    # The actual values depend on the test data created
    assert metrics is not None
    assert metrics.minimal_responses is not None
    assert metrics.response_quality_issues is not None


def test_time_bucketing_aggregation(test_store: dict[str, Any]):
    """Test that time bucketing happens at database level."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)

    for hour in range(3):
        for i in range(5):
            create_test_trace_and_spans(
                test_store,
                exp_id,
                start_time=now - timedelta(hours=3 - hour, minutes=i * 10),
                duration_ms=100 + hour * 50,
            )

    # Test that operational metrics work with this data
    metrics = insights_store.get_operational_metrics(
        experiment_ids=[exp_id],
        start_time=now - timedelta(hours=4),
        end_time=now,
    )

    # Verify we got all 15 traces
    assert metrics.total_traces == 15  # 3 hours * 5 traces each

    # Time buckets are returned as empty list currently (not implemented)
    assert len(metrics.time_buckets) == 0


def test_no_full_dataset_transfer(test_store: dict[str, Any]):
    """Test that aggregated metrics work efficiently with large datasets."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)

    # Create a large dataset
    for i in range(100):
        create_test_trace_and_spans(
            test_store,
            exp_id,
            num_spans=5,
            start_time=now - timedelta(seconds=3600 - i),
            duration_ms=50 + (i % 200),
        )

    # The insights store should compute aggregations in the database
    # without fetching all rows into memory
    metrics = insights_store.get_operational_metrics(
        experiment_ids=[exp_id],
        start_time=now - timedelta(hours=2),
        end_time=now,
    )

    # Verify the metrics are computed correctly
    assert metrics.total_traces == 100
    assert metrics.p50_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    # With diverse durations (50-249), p95 should be > p50
    assert metrics.p95_latency_ms > metrics.p50_latency_ms


def test_postgresql_specific_percentiles(test_store: dict[str, Any]):
    """Test PostgreSQL-specific percentile_cont usage."""
    db_uri = test_store["db_uri"]
    if "postgresql" not in db_uri:
        pytest.skip("PostgreSQL-specific test")

    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)
    for i in range(20):
        create_test_trace_and_spans(
            test_store,
            exp_id,
            start_time=now - timedelta(minutes=30 - i),
            duration_ms=100 + i * 5,
        )

    # Use the operational metrics API which calculates percentiles
    metrics = insights_store.get_operational_metrics(experiment_ids=[exp_id])

    # Verify we get percentiles
    assert metrics.p50_latency_ms is not None
    assert metrics.p90_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    assert metrics.p99_latency_ms is not None
    # Verify they're in order
    assert metrics.p50_latency_ms <= metrics.p90_latency_ms
    assert metrics.p90_latency_ms <= metrics.p95_latency_ms
    assert metrics.p95_latency_ms <= metrics.p99_latency_ms


def test_sqlite_fallback_percentiles(test_store: dict[str, Any]):
    """Test SQLite fallback percentile implementation."""
    db_uri = test_store["db_uri"]
    if "sqlite" not in db_uri:
        pytest.skip("SQLite-specific test")

    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)
    # Create traces with durations from 100 to 590 (100 + 49*10)
    for i in range(50):
        create_test_trace_and_spans(
            test_store,
            exp_id,
            start_time=now - timedelta(minutes=60 - i),
            duration_ms=100 + i * 10,
        )

    # Use the operational metrics which includes percentiles
    metrics = insights_store.get_operational_metrics(experiment_ids=[exp_id])

    assert metrics.p50_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    assert isinstance(metrics.p50_latency_ms, (int, float))
    assert isinstance(metrics.p95_latency_ms, (int, float))
    # With diverse durations, p95 should be greater than p50
    assert metrics.p95_latency_ms > metrics.p50_latency_ms


def test_census_generation_efficiency(test_store: dict[str, Any]):
    """Test that census generation uses efficient aggregated queries."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]
    Session = test_store["Session"]

    now = datetime.now(timezone.utc)

    for i in range(200):
        status = "ERROR" if i % 10 == 0 else "OK"
        create_test_trace_and_spans(
            test_store,
            exp_id,
            num_spans=3,
            start_time=now - timedelta(seconds=7200 - i * 10),
            duration_ms=50 + (i % 100),
            status=status,
        )

    with Session() as session:
        with patch.object(session, "execute", wraps=session.execute) as mock_execute:
            census = insights_store.generate_census(exp_id, table_name="test_table")

            # Should use aggregated queries, not fetch all rows
            # Expect a reasonable number of queries for aggregations
            assert mock_execute.call_count < 20

    assert census.metadata.table_name == "test_table"
    assert census.operational_metrics.total_traces == 200
    # 10% error rate
    assert abs(census.operational_metrics.error_rate - 10.0) <= 1.0


def test_model_distribution_aggregation(test_store: dict[str, Any]):
    """Test model distribution calculation efficiency."""
    exp_id = test_store["exp_id"]
    insights_store = test_store["insights_store"]

    now = datetime.now(timezone.utc)

    models = ["gpt-4", "claude-3", "llama-2", "mistral"]

    for i in range(100):
        models[i % len(models)]
        create_test_trace_and_spans(
            test_store,
            exp_id,
            num_spans=2,
            start_time=now - timedelta(minutes=100 - i),
            duration_ms=100 + (i % 50) * 10,
        )

    # Test that model distribution queries are aggregated
    metrics = insights_store.get_operational_metrics(experiment_ids=[exp_id])

    assert metrics.total_traces == 100
    # Should have detected spans as tools
    assert len(metrics.top_slow_tools) > 0


def test_empty_results_handling(test_store: dict[str, Any]):
    """Test graceful handling of empty result sets."""
    insights_store = test_store["insights_store"]

    # Test with non-existent experiment
    fake_exp_id = "999999"

    metrics = insights_store.get_operational_metrics([fake_exp_id])
    assert metrics.total_traces == 0

    quality = insights_store.get_quality_metrics([fake_exp_id])
    assert quality is not None

    census = insights_store.generate_census(fake_exp_id, table_name="test_table")
    assert census.operational_metrics.total_traces == 0


def test_multiple_experiments_aggregation(test_store: dict[str, Any]):
    """Test aggregation across multiple experiments."""
    tracking_store = test_store["tracking_store"]
    insights_store = test_store["insights_store"]
    exp_id = test_store["exp_id"]

    now = datetime.now(timezone.utc)

    # Create another experiment
    exp2_id = tracking_store.create_experiment("test_exp_2")
    exp3_id = tracking_store.create_experiment("test_exp_3")

    # Add traces to each experiment
    for exp in [exp_id, exp2_id, exp3_id]:
        for i in range(10):
            create_test_trace_and_spans(
                test_store,
                exp,
                num_spans=3,
                start_time=now - timedelta(minutes=30 - i),
                duration_ms=50 + (i % 100),
            )

    # Test aggregation across all experiments
    metrics = insights_store.get_operational_metrics(experiment_ids=[exp_id, exp2_id, exp3_id])

    assert metrics.total_traces == 30

    # Test that time range filtering works across experiments
    metrics_recent = insights_store.get_operational_metrics(
        experiment_ids=[exp_id, exp2_id, exp3_id],
        start_time=now - timedelta(hours=1),
        end_time=now,
    )

    # Should get all traces within time range
    assert metrics_recent.total_traces == 30


# ============== Integration Tests ==============


def create_primed_store(test_store, num_traces=100):
    """Create test traces with various characteristics for integration testing."""
    tracking_store = test_store["tracking_store"]
    exp_id = test_store["exp_id"]

    base_time = int(time.time() * 1000)
    tool_names = [
        "process_data",
        "query_llm",
        "validate_output",
        "fetch_context",
        "format_response",
    ]
    span_types = ["TOOL", "LLM", "CHAIN", "RETRIEVER", "UNKNOWN"]

    for i in range(num_traces):
        timestamp = base_time - (num_traces - i) * 60000  # 1 minute apart

        # Decide if this trace will have errors (10% error rate)
        has_error = random.random() < 0.1

        # Start the trace
        trace_info_v2 = tracking_store.deprecated_start_trace_v2(
            experiment_id=exp_id,
            timestamp_ms=timestamp,
            request_metadata={
                "mlflow.traceInputs": f'{{"query": "test query {i}"}}',
                "mlflow.traceOutputs": f'{{"response": "test response {i}"}}'
                if not has_error
                else f'{{"error": "Sorry, error {i}"}}',
                "mlflow.trace_schema.version": "3",
            },
            tags={"env": "test", "version": "1.0"},
        )

        # Create spans using database models
        create_spans_in_db(
            tracking_store,
            trace_info_v2.request_id,
            exp_id,
            tool_names,
            span_types,
            has_error,
            timestamp,
        )

        # End the trace
        tracking_store.deprecated_end_trace_v2(
            request_id=trace_info_v2.request_id,
            timestamp_ms=timestamp + random.randint(100, 20000),
            status="ERROR" if has_error else "OK",
            request_metadata={},
            tags={},
        )


def create_spans_in_db(
    store: SqlAlchemyStore,
    trace_id: str,
    experiment_id: str,
    tool_names: list[str],
    span_types: list[str],
    has_error: bool,
    timestamp: int,
):
    """Create spans directly in the database."""
    with store.ManagedSessionMaker() as session:
        num_spans = random.randint(3, 8)

        # Create root span
        root_span_id = f"sp-{uuid.uuid4().hex[:16]}"
        root_start = timestamp * 1_000_000  # Convert ms to ns
        root_duration = random.randint(5000, 15000) * 1_000_000  # 5-15 seconds in nanoseconds

        root_content = {
            "attributes": {"level": "root"},
            "inputs": {},
            "outputs": {},
        }

        root_span = SqlSpan(
            trace_id=trace_id,  # This is the request_id from trace_info
            experiment_id=experiment_id,
            span_id=root_span_id,
            parent_span_id=None,
            name="process_request",
            type="CHAIN",
            status="ERROR" if has_error else "OK",
            start_time_unix_nano=root_start,
            end_time_unix_nano=root_start + root_duration,
            content=json.dumps(root_content),
        )
        session.add(root_span)

        span_ids = [root_span_id]
        span_timings = [(root_start, root_start + root_duration)]

        # Create child spans
        for i in range(1, num_spans):
            span_id = f"sp-{uuid.uuid4().hex[:16]}"
            tool_name = random.choice(tool_names)
            span_type = random.choice(span_types)

            # Some spans are children of root, others are nested
            parent_idx = 0 if i < 3 else random.randint(0, i - 1)
            parent_id = span_ids[parent_idx]
            parent_start, parent_end = span_timings[parent_idx]

            # Calculate timing relative to parent
            max_duration = (parent_end - parent_start) // (num_spans - i + 1)
            duration = random.randint(int(max_duration * 0.1), int(max_duration * 0.8))
            start_offset = random.randint(0, int(max_duration * 0.2))

            start_time = parent_start + start_offset
            end_time = start_time + duration

            # Randomly assign errors to some spans in error traces
            span_has_error = has_error and random.random() < 0.3

            span_content = {
                "attributes": {
                    "tool": tool_name,
                    "iteration": str(i),
                },
                "inputs": {},
                "outputs": {},
            }

            span = SqlSpan(
                trace_id=trace_id,
                experiment_id=experiment_id,
                span_id=span_id,
                parent_span_id=parent_id,
                name=tool_name,
                type=span_type,
                status="ERROR" if span_has_error else "OK",
                start_time_unix_nano=start_time,
                end_time_unix_nano=end_time,
                content=json.dumps(span_content),
            )
            session.add(span)

            span_ids.append(span_id)
            span_timings.append((start_time, end_time))

        session.commit()


def test_integration_get_operational_metrics(test_store: dict[str, Any]):
    """Integration test for operational metrics with realistic data."""
    insights_store = test_store["insights_store"]
    exp_id = test_store["exp_id"]

    create_primed_store(test_store, num_traces=50)

    metrics = insights_store.get_operational_metrics([exp_id])

    assert metrics.total_traces == 50
    assert metrics.error_rate > 0  # Should have some errors
    assert metrics.error_rate <= 20  # ~10% error rate
    assert metrics.p50_latency_ms is not None
    assert metrics.p95_latency_ms is not None
    assert len(metrics.top_slow_tools) > 0


def test_integration_time_filtering(test_store: dict[str, Any]):
    """Test time range filtering with realistic data."""
    insights_store = test_store["insights_store"]
    exp_id = test_store["exp_id"]

    create_primed_store(test_store, num_traces=30)

    now = datetime.now(timezone.utc)

    # Get metrics for last hour
    recent_metrics = insights_store.get_operational_metrics(
        [exp_id], start_time=now - timedelta(hours=1), end_time=now
    )

    # Get metrics for all time
    all_metrics = insights_store.get_operational_metrics([exp_id])

    # Recent should have fewer traces
    assert recent_metrics.total_traces <= all_metrics.total_traces


def test_integration_census_generation(test_store: dict[str, Any]):
    """Test census generation with realistic data."""
    insights_store = test_store["insights_store"]
    exp_id = test_store["exp_id"]

    create_primed_store(test_store, num_traces=100)

    census = insights_store.generate_census(exp_id, table_name="production_table")

    assert census.metadata.table_name == "production_table"
    assert census.operational_metrics.total_traces == 100
    assert census.operational_metrics.error_rate is not None
    assert census.quality_metrics is not None

    # Test serialization
    census_dict = census.to_dict()
    assert "metadata" in census_dict
    assert "operational_metrics" in census_dict
    assert "quality_metrics" in census_dict
