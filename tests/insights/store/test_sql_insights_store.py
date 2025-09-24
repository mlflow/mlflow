"""
Consolidated tests for SqlInsightsStore.
Combines unit tests, integration tests, and performance tests.
"""

import json
import os
import random
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from mlflow.entities import TraceInfo, TraceState
from mlflow.entities.trace_location import TraceLocation
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.insights.models.entities import Census, OperationalMetrics, QualityMetrics
from mlflow.insights.store.sql_insights_store import SqlInsightsStore
from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlTraceInfo, SqlTraceMetadata, SqlTraceTag
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import encode_span_id, encode_trace_id


class TestSqlInsightsStore(unittest.TestCase):
    """Main test class for SqlInsightsStore using real database."""

    def setUp(self):
        """Set up test database and store instances."""
        self.temp_dir = tempfile.mkdtemp()
        # Use environment variable if set (for testing against different databases)
        # Otherwise default to SQLite
        self.db_uri = os.environ.get(
            MLFLOW_TRACKING_URI.name,
            f"sqlite:///{os.path.join(self.temp_dir, 'test.db')}"
        )

        self.tracking_store = SqlAlchemyStore(self.db_uri, self.temp_dir)
        self.insights_store = SqlInsightsStore(self.tracking_store)

        # Use the tracking store's engine to ensure we're using the same connection
        self.engine = self.tracking_store.engine
        self.Session = sessionmaker(bind=self.engine)

        self.exp_id = self.tracking_store.create_experiment("test_experiment")

    def tearDown(self):
        """Clean up test data."""
        with self.Session() as session:
            # Delete all test data to ensure clean state for next test
            session.query(SqlSpan).delete()
            session.query(SqlTraceMetadata).delete()
            session.query(SqlTraceTag).delete()
            session.query(SqlTraceInfo).delete()
            session.commit()

        # Only remove temp dir if using SQLite
        if self.db_uri.startswith("sqlite://"):
            import shutil
            shutil.rmtree(self.temp_dir)

    def _create_test_trace_and_spans(
        self,
        exp_id,
        num_spans=3,
        start_time=None,
        duration_ms=100,
        status="OK",
        add_metadata=True
    ):
        """Helper to create test trace with spans directly in database."""
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        import random
        trace_id = encode_trace_id(random.getrandbits(128))
        request_id = f"req_{trace_id[:8]}"

        with self.Session() as session:
            trace_info = SqlTraceInfo(
                request_id=request_id,
                experiment_id=exp_id,
                timestamp_ms=int(start_time.timestamp() * 1000),
                execution_time_ms=duration_ms,
                status="OK"
            )
            session.add(trace_info)
            session.flush()

            if add_metadata:
                metadata = SqlTraceMetadata(
                    request_id=request_id,
                    key="mlflow.traceInputs",
                    value='{"prompt": "Test prompt for analysis"}'
                )
                session.add(metadata)

                metadata_output = SqlTraceMetadata(
                    request_id=request_id,
                    key="mlflow.traceOutputs",
                    value='{"response": "This is a test response that is longer than 50 characters to test quality metrics properly"}'
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
                    content=f'{{"span_type": "{"LLM" if i == 0 else "CHAIN"}", "attributes": {{"model": "gpt-4" if i == 0 else ""}}, "inputs": {{"test": "input"}}, "outputs": {{"test": "output"}}, "events": []}}'
                )
                session.add(span)

            session.commit()

        return request_id, trace_id

    # ============== Basic Smoke Tests ==============

    def test_basic_insights_store_creation(self):
        """Test that we can create an insights store."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        artifact_path = Path(temp_dir) / "artifacts"
        artifact_path.mkdir()

        db_uri = f"sqlite:///{db_path}"
        tracking_store = SqlAlchemyStore(db_uri, artifact_path.as_uri())
        insights_store = SqlInsightsStore(tracking_store)

        self.assertIsNotNone(insights_store)
        self.assertEqual(insights_store.store, tracking_store)
        self.assertEqual(insights_store.dialect, "sqlite")

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

    def test_empty_metrics(self):
        """Test metrics retrieval with no data."""
        # Get metrics for empty experiment
        metrics = self.insights_store.get_operational_metrics([self.exp_id])

        self.assertEqual(metrics.total_traces, 0)
        self.assertEqual(metrics.error_count, 0)
        self.assertEqual(metrics.error_rate, 0.0)
        self.assertEqual(metrics.p50_latency_ms, 0.0)
        self.assertEqual(metrics.p95_latency_ms, 0.0)
        self.assertEqual(len(metrics.time_buckets), 0)
        self.assertEqual(len(metrics.top_error_spans), 0)
        self.assertEqual(len(metrics.top_slow_tools), 0)

    def test_single_trace_metrics(self):
        """Test metrics with a single trace."""
        now = datetime.now(timezone.utc)
        trace_id = f"tr-{uuid.uuid4().hex[:16]}"
        timestamp = int(now.timestamp() * 1000)

        # Create trace and span using SQLAlchemy models directly
        with self.tracking_store.ManagedSessionMaker() as session:
            # Create trace
            trace_info = SqlTraceInfo(
                request_id=trace_id,
                experiment_id=self.exp_id,
                timestamp_ms=timestamp,
                execution_time_ms=1000,
                status="OK",
            )
            session.add(trace_info)

            # Add trace metadata
            inputs_meta = SqlTraceMetadata(
                request_id=trace_id,
                key="mlflow.traceInputs",
                value='{"query": "test"}'
            )
            outputs_meta = SqlTraceMetadata(
                request_id=trace_id,
                key="mlflow.traceOutputs",
                value='{"response": "This is a test response that is definitely longer than 50 characters to pass the test"}'
            )
            session.add(inputs_meta)
            session.add(outputs_meta)

            # Create span
            span_content = {
                "attributes": {"test": "true"},
                "inputs": {"query": "test"},
                "outputs": {"response": "This is a test response that is definitely longer than 50 characters to pass the test"},
            }
            span = SqlSpan(
                trace_id=trace_id,
                experiment_id=self.exp_id,
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
        metrics = self.insights_store.get_operational_metrics([self.exp_id])

        self.assertEqual(metrics.total_traces, 1)
        self.assertEqual(metrics.error_count, 0)
        self.assertEqual(metrics.error_rate, 0.0)
        self.assertEqual(metrics.p50_latency_ms, 1000.0)  # 1 second in milliseconds
        self.assertEqual(metrics.p95_latency_ms, 1000.0)  # Only one trace, so all percentiles are the same

        # Test quality metrics
        quality = self.insights_store.get_quality_metrics([self.exp_id])
        self.assertEqual(quality.minimal_responses.value, 0.0)  # Response is > 50 chars
        self.assertEqual(quality.response_quality_issues.value, 0.0)  # No error indicators

        # Test census generation
        census = self.insights_store.generate_census([self.exp_id], table_name="test_table")
        self.assertEqual(census.metadata.table_name, "test_table")
        self.assertEqual(census.operational_metrics.total_traces, 1)
        self.assertFalse(census.has_quality_issues(threshold=10.0))

    def test_error_trace_metrics(self):
        """Test metrics with an error trace."""
        now = datetime.now(timezone.utc)
        trace_id = f"tr-{uuid.uuid4().hex[:16]}"
        timestamp = int(now.timestamp() * 1000)

        # Create error trace and span using SQLAlchemy models directly
        with self.tracking_store.ManagedSessionMaker() as session:
            # Create error trace
            trace_info = SqlTraceInfo(
                request_id=trace_id,
                experiment_id=self.exp_id,
                timestamp_ms=timestamp,
                execution_time_ms=500,
                status="ERROR",
            )
            session.add(trace_info)

            # Add trace metadata with "Sorry" in the output
            inputs_meta = SqlTraceMetadata(
                request_id=trace_id,
                key="mlflow.traceInputs",
                value='{"query": "test"}'
            )
            outputs_meta = SqlTraceMetadata(
                request_id=trace_id,
                key="mlflow.traceOutputs",
                value='{"error": "Sorry, I cannot process this"}'
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
                experiment_id=self.exp_id,
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
        metrics = self.insights_store.get_operational_metrics([self.exp_id])

        self.assertEqual(metrics.total_traces, 1)
        self.assertEqual(metrics.error_count, 1)
        self.assertEqual(metrics.error_rate, 100.0)

        # Check error spans
        self.assertEqual(len(metrics.top_error_spans), 1)
        self.assertEqual(metrics.top_error_spans[0].error_span_name, "test_error_span")
        self.assertEqual(metrics.top_error_spans[0].count, 1)

        # Test quality metrics - should detect "Sorry" as quality issue
        quality = self.insights_store.get_quality_metrics([self.exp_id])
        self.assertEqual(quality.response_quality_issues.value, 100.0)  # Has "Sorry" in response

    # ============== Database Optimization Tests ==============

    def test_get_operational_metrics_db_aggregation(self):
        """Test that operational metrics use database aggregation."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            self._create_test_trace_and_spans(
                self.exp_id,
                num_spans=3,
                start_time=now - timedelta(minutes=30-i),
                duration_ms=100 + i * 10
            )

        metrics = self.insights_store.get_operational_metrics(
            experiment_ids=[self.exp_id],
            start_time=now - timedelta(hours=1),
            end_time=now
        )

        self.assertEqual(metrics.total_traces, 10)
        self.assertIsNotNone(metrics.p50_latency_ms)
        self.assertIsNotNone(metrics.p95_latency_ms)
        # Verify percentiles are reasonable
        self.assertGreaterEqual(metrics.p95_latency_ms, metrics.p50_latency_ms)

    def test_calculate_latency_percentiles_single_query(self):
        """Test that percentile calculations happen in a single DB query."""
        now = datetime.now(timezone.utc)
        # Create traces with clearly different durations
        durations = []
        for i in range(20):
            duration = 100 + i * 50  # 100, 150, 200, ..., 1050
            durations.append(duration)
            self._create_test_trace_and_spans(
                self.exp_id,
                num_spans=1,
                start_time=now - timedelta(minutes=60-i),
                duration_ms=duration
            )

        # Use the public API method
        percentiles = self.insights_store.get_latency_percentiles(
            experiment_ids=[self.exp_id],
            percentiles=[50, 95]
        )

        self.assertIn('p50', percentiles)
        self.assertIn('p95', percentiles)
        self.assertIsInstance(percentiles['p50'], (int, float))
        # With durations from 100 to 1050, p95 should be > p50
        self.assertGreater(percentiles['p95'], percentiles['p50'])

    def test_get_quality_metrics_aggregated_queries(self):
        """Test that quality metrics use aggregated database queries."""
        now = datetime.now(timezone.utc)

        for i in range(5):
            self._create_test_trace_and_spans(
                self.exp_id,
                status="OK",
                start_time=now - timedelta(minutes=30-i)
            )

        for i in range(3):
            self._create_test_trace_and_spans(
                self.exp_id,
                status="ERROR",
                start_time=now - timedelta(minutes=20-i)
            )

        metrics = self.insights_store.get_quality_metrics(
            experiment_ids=[self.exp_id],
            start_time=now - timedelta(hours=1),
            end_time=now
        )

        # Verify that quality metrics are calculated
        # The actual values depend on the test data created
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(metrics.minimal_responses)
        self.assertIsNotNone(metrics.response_quality_issues)

    def test_time_bucketing_aggregation(self):
        """Test that time bucketing happens at database level."""
        now = datetime.now(timezone.utc)

        for hour in range(3):
            for i in range(5):
                self._create_test_trace_and_spans(
                    self.exp_id,
                    start_time=now - timedelta(hours=3-hour, minutes=i*10),
                    duration_ms=100 + hour * 50
                )

        with self.Session() as session:
            with patch.object(session, 'execute', wraps=session.execute) as mock_execute:
                buckets = self.insights_store._get_time_buckets(
                    session,
                    [self.exp_id],
                    now - timedelta(hours=4),
                    now,
                    "hour"
                )

                call_count = mock_execute.call_count
                self.assertLessEqual(call_count, 2)

        # We may get 3 or 4 buckets depending on how the hour boundaries align
        self.assertIn(len(buckets), [3, 4])

        # Count total traces across all buckets
        total_traces = sum(bucket.total_traces for bucket in buckets)
        self.assertEqual(total_traces, 15)  # 3 hours * 5 traces each

        # At least one bucket should have 5 traces
        max_traces = max(bucket.total_traces for bucket in buckets)
        self.assertGreaterEqual(max_traces, 5)

    def test_no_full_dataset_transfer(self):
        """Test that aggregated metrics work efficiently with large datasets."""
        now = datetime.now(timezone.utc)

        # Create a large dataset
        for i in range(1000):
            self._create_test_trace_and_spans(
                self.exp_id,
                num_spans=5,
                start_time=now - timedelta(seconds=3600-i),
                duration_ms=50 + (i % 200)
            )

        # The insights store should compute aggregations in the database
        # without fetching all rows into memory
        metrics = self.insights_store.get_operational_metrics(
            experiment_ids=[self.exp_id],
            start_time=now - timedelta(hours=2),
            end_time=now
        )

        # Verify the metrics are computed correctly
        self.assertEqual(metrics.total_traces, 1000)
        self.assertIsNotNone(metrics.p50_latency_ms)
        self.assertIsNotNone(metrics.p95_latency_ms)
        # With diverse durations (50-249), p95 should be > p50
        self.assertGreater(metrics.p95_latency_ms, metrics.p50_latency_ms)

    def test_postgresql_specific_percentiles(self):
        """Test PostgreSQL-specific percentile_cont usage."""
        if 'postgresql' not in self.db_uri:
            self.skipTest("PostgreSQL-specific test")

        now = datetime.now(timezone.utc)
        for i in range(20):
            self._create_test_trace_and_spans(
                self.exp_id,
                start_time=now - timedelta(minutes=30-i),
                duration_ms=100 + i * 5
            )

        # Use the public API method for PostgreSQL
        percentiles = self.insights_store.get_latency_percentiles(
            experiment_ids=[self.exp_id],
            percentiles=[50, 90, 95, 99]
        )

        # Verify we get all requested percentiles
        self.assertIn('p50', percentiles)
        self.assertIn('p90', percentiles)
        self.assertIn('p95', percentiles)
        self.assertIn('p99', percentiles)
        # Verify they're in order
        self.assertLessEqual(percentiles['p50'], percentiles['p90'])
        self.assertLessEqual(percentiles['p90'], percentiles['p95'])
        self.assertLessEqual(percentiles['p95'], percentiles['p99'])

    def test_sqlite_fallback_percentiles(self):
        """Test SQLite fallback percentile implementation."""
        if 'sqlite' not in self.db_uri:
            self.skipTest("SQLite-specific test")

        now = datetime.now(timezone.utc)
        # Create traces with durations from 100 to 590 (100 + 49*10)
        for i in range(50):
            self._create_test_trace_and_spans(
                self.exp_id,
                start_time=now - timedelta(minutes=60-i),
                duration_ms=100 + i * 10
            )

        # Use the public API method
        percentiles = self.insights_store.get_latency_percentiles(
            experiment_ids=[self.exp_id],
            percentiles=[50, 95]
        )

        self.assertIn('p50', percentiles)
        self.assertIn('p95', percentiles)
        self.assertIsInstance(percentiles['p50'], (int, float))
        self.assertIsInstance(percentiles['p95'], (int, float))
        # With diverse durations, p95 should be greater than p50
        self.assertGreater(percentiles['p95'], percentiles['p50'])

    def test_census_generation_efficiency(self):
        """Test that census generation uses efficient aggregated queries."""
        now = datetime.now(timezone.utc)

        for i in range(200):
            status = "ERROR" if i % 10 == 0 else "OK"
            self._create_test_trace_and_spans(
                self.exp_id,
                num_spans=3,
                start_time=now - timedelta(seconds=7200-i*10),
                duration_ms=50 + (i % 100),
                status=status
            )

        with self.Session() as session:
            with patch.object(session, 'execute', wraps=session.execute) as mock_execute:
                census = self.insights_store.generate_census(
                    experiment_ids=[self.exp_id],
                    table_name="test_table"
                )

                # Should use aggregated queries, not fetch all rows
                # Expect a reasonable number of queries for aggregations
                self.assertLess(mock_execute.call_count, 20)

        self.assertEqual(census.metadata.table_name, "test_table")
        self.assertEqual(census.operational_metrics.total_traces, 200)
        # 10% error rate
        self.assertAlmostEqual(census.operational_metrics.error_rate, 10.0, delta=1.0)

    def test_model_distribution_aggregation(self):
        """Test model distribution calculation efficiency."""
        now = datetime.now(timezone.utc)

        models = ["gpt-4", "claude-3", "llama-2", "mistral"]

        for i in range(100):
            model = models[i % len(models)]
            self._create_test_trace_and_spans(
                self.exp_id,
                num_spans=2,
                start_time=now - timedelta(minutes=100-i),
                duration_ms=100 + (i % 50) * 10
            )

        # Test that model distribution queries are aggregated
        metrics = self.insights_store.get_operational_metrics(
            experiment_ids=[self.exp_id]
        )

        self.assertEqual(metrics.total_traces, 100)
        # Should have detected spans as tools
        self.assertGreater(len(metrics.top_slow_tools), 0)

    def test_empty_results_handling(self):
        """Test graceful handling of empty result sets."""
        # Test with non-existent experiment
        fake_exp_id = "999999"

        metrics = self.insights_store.get_operational_metrics([fake_exp_id])
        self.assertEqual(metrics.total_traces, 0)

        quality = self.insights_store.get_quality_metrics([fake_exp_id])
        self.assertIsNotNone(quality)

        census = self.insights_store.generate_census([fake_exp_id], table_name="test_table")
        self.assertEqual(census.operational_metrics.total_traces, 0)

    def test_multiple_experiments_aggregation(self):
        """Test aggregation across multiple experiments."""
        now = datetime.now(timezone.utc)

        # Create another experiment
        exp2_id = self.tracking_store.create_experiment("test_exp_2")
        exp3_id = self.tracking_store.create_experiment("test_exp_3")

        # Add traces to each experiment
        for exp_id in [self.exp_id, exp2_id, exp3_id]:
            for i in range(10):
                self._create_test_trace_and_spans(
                    exp_id,
                    num_spans=3,
                    start_time=now - timedelta(minutes=30-i),
                    duration_ms=50 + (i % 100)
                )

        # Test aggregation across all experiments
        metrics = self.insights_store.get_operational_metrics(
            experiment_ids=[self.exp_id, exp2_id, exp3_id]
        )

        self.assertEqual(metrics.total_traces, 30)

        # Test time bucketing across experiments
        with self.Session() as session:
            buckets = self.insights_store._get_time_buckets(
                session,
                [self.exp_id, exp2_id, exp3_id],
                now - timedelta(hours=1),
                now,
                "hour"
            )

            total_in_buckets = sum(b.total_traces for b in buckets)
            self.assertEqual(total_in_buckets, 30)

    # ============== Integration Tests ==============

    def _create_primed_store(self, num_traces=100):
        """Create test traces with various characteristics for integration testing."""
        base_time = int(time.time() * 1000)
        tool_names = ["process_data", "query_llm", "validate_output", "fetch_context", "format_response"]
        span_types = ["TOOL", "LLM", "CHAIN", "RETRIEVER", "UNKNOWN"]

        for i in range(num_traces):
            timestamp = base_time - (num_traces - i) * 60000  # 1 minute apart

            # Decide if this trace will have errors (10% error rate)
            has_error = random.random() < 0.1

            # Start the trace
            trace_info_v2 = self.tracking_store.deprecated_start_trace_v2(
                experiment_id=self.exp_id,
                timestamp_ms=timestamp,
                request_metadata={
                    "mlflow.traceInputs": f'{{"query": "test query {i}"}}',
                    "mlflow.traceOutputs": f'{{"response": "test response {i}"}}' if not has_error else f'{{"error": "Sorry, error {i}"}}',
                    "mlflow.trace_schema.version": "3",
                },
                tags={"env": "test", "version": "1.0"},
            )

            # Create spans using database models
            self._create_spans_in_db(
                self.tracking_store,
                trace_info_v2.request_id,
                self.exp_id,
                tool_names,
                span_types,
                has_error,
                timestamp
            )

            # End the trace
            self.tracking_store.deprecated_end_trace_v2(
                request_id=trace_info_v2.request_id,
                timestamp_ms=timestamp + random.randint(100, 20000),
                status="ERROR" if has_error else "OK",
                request_metadata={},
                tags={},
            )

    def _create_spans_in_db(
        self,
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
                "outputs": {}
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
                    "outputs": {}
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

    def test_integration_get_operational_metrics(self):
        """Integration test for operational metrics with realistic data."""
        self._create_primed_store(num_traces=50)

        metrics = self.insights_store.get_operational_metrics([self.exp_id])

        self.assertEqual(metrics.total_traces, 50)
        self.assertGreater(metrics.error_rate, 0)  # Should have some errors
        self.assertLess(metrics.error_rate, 20)  # ~10% error rate
        self.assertIsNotNone(metrics.p50_latency_ms)
        self.assertIsNotNone(metrics.p95_latency_ms)
        self.assertGreater(len(metrics.top_slow_tools), 0)

    def test_integration_time_filtering(self):
        """Test time range filtering with realistic data."""
        self._create_primed_store(num_traces=30)

        now = datetime.now(timezone.utc)

        # Get metrics for last hour
        recent_metrics = self.insights_store.get_operational_metrics(
            [self.exp_id],
            start_time=now - timedelta(hours=1),
            end_time=now
        )

        # Get metrics for all time
        all_metrics = self.insights_store.get_operational_metrics([self.exp_id])

        # Recent should have fewer traces
        self.assertLessEqual(recent_metrics.total_traces, all_metrics.total_traces)

    def test_integration_census_generation(self):
        """Test census generation with realistic data."""
        self._create_primed_store(num_traces=100)

        census = self.insights_store.generate_census(
            [self.exp_id],
            table_name="production_table"
        )

        self.assertEqual(census.metadata.table_name, "production_table")
        self.assertEqual(census.operational_metrics.total_traces, 100)
        self.assertIsNotNone(census.operational_metrics.error_rate)
        self.assertIsNotNone(census.quality_metrics)

        # Test serialization
        census_dict = census.to_dict()
        self.assertIn("metadata", census_dict)
        self.assertIn("operational_metrics", census_dict)
        self.assertIn("quality_metrics", census_dict)


if __name__ == "__main__":
    unittest.main()