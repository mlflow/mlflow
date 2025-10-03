"""
Tests for DatabricksSqlInsightsStore.

This test suite mocks the Databricks REST API calls and SQL query execution
to test the insights store functionality without requiring a real Databricks workspace.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.insights.models.entities import (
    Census,
)
from mlflow.insights.store.databricks_sql_insights_store import DatabricksSqlInsightsStore


@pytest.fixture
def mock_store():
    """Set up test fixtures with mocked Databricks store."""
    # Mock the parent DatabricksSqlStore initialization
    with patch(
        "mlflow.insights.store.databricks_sql_insights_store.DatabricksSqlStore.__init__"
    ) as mock_init:
        mock_init.return_value = None
        store = DatabricksSqlInsightsStore(store_uri="databricks")

    # Add required attributes that parent would normally set
    store._spark_session = None
    store.get_host_creds = MagicMock()

    # Mock the REST API methods
    store.perform_request = MagicMock()
    store.execute_sql = MagicMock()
    store._get_trace_table_for_experiment = MagicMock()

    return store


# ============== Basic Unit Tests ==============


def test_generate_census_empty_experiment(mock_store):
    """Test census generation with no data."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    # Mock empty trace table
    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock SQL query results - all empty
    mock_store.execute_sql.side_effect = [
        # Basic counts query
        [{"total_traces": 0, "ok_count": 0, "error_count": 0, "error_rate_percentage": 0.0}],
        # Latency percentiles query
        [
            {
                "p50_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            }
        ],
        # Error spans query
        [],
        # Slow tools query
        [],
        # Timestamp range query
        [
            {
                "first_trace_timestamp": datetime.now(timezone.utc).isoformat(),
                "last_trace_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        # Quality metrics queries (4 of them)
        [{"verbose_percentage": 0.0, "sample_trace_ids": []}],
        [{"rushed_complex_pct": 0.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 0.0, "sample_trace_ids": []}],
        [{"problematic_response_rate_percentage": 0.0, "sample_trace_ids": []}],
    ]

    # Generate census
    census = mock_store.generate_census(experiment_id)

    # Verify census structure
    assert isinstance(census, Census)
    assert census.operational_metrics.total_traces == 0
    assert census.operational_metrics.error_count == 0
    assert census.operational_metrics.error_rate == 0.0
    assert len(census.operational_metrics.top_error_spans) == 0
    assert len(census.operational_metrics.top_slow_tools) == 0


def test_generate_census_with_traces(mock_store):
    """Test census generation with trace data."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock SQL query results with actual data
    mock_store.execute_sql.side_effect = [
        # Basic counts query
        [
            {
                "total_traces": 100,
                "ok_count": 90,
                "error_count": 10,
                "error_rate_percentage": 10.0,
            }
        ],
        # Latency percentiles query
        [
            {
                "p50_latency_ms": 100.0,
                "p90_latency_ms": 500.0,
                "p95_latency_ms": 750.0,
                "p99_latency_ms": 1000.0,
                "max_latency_ms": 2000.0,
            }
        ],
        # Error spans query
        [
            {
                "error_span_name": "api_call",
                "count": 5,
                "percentage_of_errors": 50.0,
                "sample_trace_ids": ["trace1", "trace2"],
            },
            {
                "error_span_name": "db_query",
                "count": 3,
                "percentage_of_errors": 30.0,
                "sample_trace_ids": ["trace3"],
            },
        ],
        # Slow tools query
        [
            {
                "tool_span_name": "llm_invoke",
                "count": 20,
                "median_latency_ms": 300.0,
                "p95_latency_ms": 800.0,
                "sample_trace_ids": ["trace4", "trace5"],
            }
        ],
        # Timestamp range query
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-02T00:00:00Z",
            }
        ],
        # Quality metrics queries
        [{"verbose_percentage": 15.5, "sample_trace_ids": ["trace6", "trace7"]}],
        [{"rushed_complex_pct": 8.2, "sample_trace_ids": ["trace8"]}],
        [{"minimal_response_rate": 5.0, "sample_trace_ids": ["trace9"]}],
        [
            {
                "problematic_response_rate_percentage": 12.0,
                "sample_trace_ids": ["trace10", "trace11"],
            }
        ],
    ]

    # Generate census
    census = mock_store.generate_census(experiment_id)

    # Verify operational metrics
    assert census.operational_metrics.total_traces == 100
    assert census.operational_metrics.ok_count == 90
    assert census.operational_metrics.error_count == 10
    assert census.operational_metrics.error_rate == 10.0
    assert census.operational_metrics.p50_latency_ms == 100.0
    assert census.operational_metrics.p95_latency_ms == 750.0

    # Verify error spans
    assert len(census.operational_metrics.top_error_spans) == 2
    assert census.operational_metrics.top_error_spans[0].error_span_name == "api_call"
    assert census.operational_metrics.top_error_spans[0].count == 5

    # Verify slow tools
    assert len(census.operational_metrics.top_slow_tools) == 1
    assert census.operational_metrics.top_slow_tools[0].tool_span_name == "llm_invoke"

    # Verify quality metrics
    assert census.quality_metrics.verbosity.value == 15.5
    assert census.quality_metrics.rushed_processing.value == 8.2
    assert census.quality_metrics.minimal_responses.value == 5.0
    assert census.quality_metrics.response_quality_issues.value == 12.0


def test_generate_census_no_trace_table(mock_store):
    """Test census generation when no trace table exists for experiment."""
    experiment_id = "12345"
    mock_store._get_trace_table_for_experiment.return_value = None

    with pytest.raises(MlflowException, match="No trace table found") as context:
        mock_store.generate_census(experiment_id)
    # MlflowException error_code can be string or integer
    assert context.value.error_code in ["INVALID_PARAMETER_VALUE", 1000]


def test_generate_census_invalid_experiment_id(mock_store):
    """Test census generation with invalid experiment ID."""
    with pytest.raises(MlflowException, match="experiment_id parameter is required"):
        mock_store.generate_census("")


def test_generate_census_with_sample_trace_cleanup(mock_store):
    """Test that sample trace IDs are properly cleaned (None values filtered)."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock results with None values in sample_trace_ids
    mock_store.execute_sql.side_effect = [
        # Basic counts
        [{"total_traces": 10, "ok_count": 10, "error_count": 0, "error_rate_percentage": 0.0}],
        # Latency percentiles
        [
            {
                "p50_latency_ms": 100.0,
                "p90_latency_ms": 200.0,
                "p95_latency_ms": 250.0,
                "p99_latency_ms": 300.0,
                "max_latency_ms": 400.0,
            }
        ],
        # Error spans - with None values
        [
            {
                "error_span_name": "test_span",
                "count": 1,
                "percentage_of_errors": 100.0,
                "sample_trace_ids": [None, None, "trace2", None, "trace3", None],
            }
        ],
        # Slow tools - with None values
        [
            {
                "tool_span_name": "test_tool",
                "count": 5,
                "median_latency_ms": 150.0,
                "p95_latency_ms": 250.0,
                "sample_trace_ids": ["trace4", None, None, "trace5"],
            }
        ],
        # Timestamps
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-01T01:00:00Z",
            }
        ],
        # Quality metrics with None values
        [{"verbose_percentage": 0.0, "sample_trace_ids": [None, None]}],
        [{"rushed_complex_pct": 0.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 0.0, "sample_trace_ids": [None, "trace6", None]}],
        [{"problematic_response_rate_percentage": 0.0, "sample_trace_ids": []}],
    ]

    census = mock_store.generate_census(experiment_id)

    # Verify None values are filtered out
    error_span = census.operational_metrics.top_error_spans[0]
    assert error_span.sample_trace_ids == ["trace2", "trace3"]

    slow_tool = census.operational_metrics.top_slow_tools[0]
    assert slow_tool.sample_trace_ids == ["trace4", "trace5"]

    # Check quality metrics sample traces
    assert census.quality_metrics.minimal_responses.sample_trace_ids == ["trace6"]
    assert census.quality_metrics.verbosity.sample_trace_ids == []


def test_census_serialization(mock_store):
    """Test that census can be serialized to JSON."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock minimal valid response
    mock_store.execute_sql.side_effect = [
        [{"total_traces": 1, "ok_count": 1, "error_count": 0, "error_rate_percentage": 0.0}],
        [
            {
                "p50_latency_ms": 100.0,
                "p90_latency_ms": 100.0,
                "p95_latency_ms": 100.0,
                "p99_latency_ms": 100.0,
                "max_latency_ms": 100.0,
            }
        ],
        [],  # No error spans
        [],  # No slow tools
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-01T00:00:00Z",
            }
        ],
        [{"verbose_percentage": 0.0, "sample_trace_ids": []}],
        [{"rushed_complex_pct": 0.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 0.0, "sample_trace_ids": []}],
        [{"problematic_response_rate_percentage": 0.0, "sample_trace_ids": []}],
    ]

    census = mock_store.generate_census(experiment_id)

    # Test JSON serialization
    census_json = census.model_dump_json(indent=2)
    assert isinstance(census_json, str)

    # Parse JSON to verify structure
    census_dict = json.loads(census_json)
    assert "metadata" in census_dict
    assert "operational_metrics" in census_dict
    assert "quality_metrics" in census_dict

    # Test dict conversion
    census_dict = census.to_dict()
    assert isinstance(census_dict, dict)
    assert census_dict["operational_metrics"]["total_traces"] == 1


def test_execute_sql_error_handling(mock_store):
    """Test error handling when SQL execution fails."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock SQL execution failure
    mock_store.execute_sql.side_effect = Exception("SQL execution failed")

    with pytest.raises(Exception, match="SQL execution failed"):
        mock_store.generate_census(experiment_id)


def test_get_trace_table_for_experiment_implementation(mock_store):
    """Test the _get_trace_table_for_experiment method behavior through mocking."""
    # Since _get_trace_table_for_experiment is implemented in the parent class
    # DatabricksSqlStore, and we're mocking it in our fixture, we'll test
    # that our code properly calls this method with the right experiment_id

    mock_store._get_trace_table_for_experiment.return_value = "catalog.schema.traces_12345"

    # Trigger a call that uses _get_trace_table_for_experiment
    mock_store.execute_sql.side_effect = [
        [{"total_traces": 1, "ok_count": 1, "error_count": 0, "error_rate_percentage": 0.0}],
        [
            {
                "p50_latency_ms": 100.0,
                "p90_latency_ms": 100.0,
                "p95_latency_ms": 100.0,
                "p99_latency_ms": 100.0,
                "max_latency_ms": 100.0,
            }
        ],
        [],
        [],
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-01T00:00:00Z",
            }
        ],
        [{"verbose_percentage": 0.0, "sample_trace_ids": []}],
        [{"rushed_complex_pct": 0.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 0.0, "sample_trace_ids": []}],
        [{"problematic_response_rate_percentage": 0.0, "sample_trace_ids": []}],
    ]

    # Generate census which should call _get_trace_table_for_experiment
    mock_store.generate_census("test_exp_123")

    # Verify the method was called with the correct experiment_id
    mock_store._get_trace_table_for_experiment.assert_called_once_with("test_exp_123")


def test_get_trace_table_no_tracing_enabled():
    """Test when experiment doesn't have tracing enabled."""
    # Mock REST API response without tracing tags
    mock_response = {"experiment": {"tags": [{"key": "some.other.tag", "value": "value"}]}}

    with patch(
        "mlflow.insights.store.databricks_sql_insights_store.DatabricksSqlStore.__init__"
    ) as mock_init:
        mock_init.return_value = None
        store = DatabricksSqlInsightsStore(store_uri="databricks")
        store._spark_session = None
        store.get_host_creds = MagicMock()
        store.perform_request = MagicMock(return_value=mock_response)

        table_name = store._get_trace_table_for_experiment("12345")

        assert table_name is None


def test_complex_error_span_aggregation(mock_store):
    """Test aggregation of complex error span data."""
    experiment_id = "12345"
    table_name = "traces_table_12345"

    mock_store._get_trace_table_for_experiment.return_value = table_name

    # Mock complex error span results
    error_spans = [
        {
            "error_span_name": "api_call_timeout",
            "count": 25,
            "percentage_of_errors": 35.7,
            "sample_trace_ids": ["t1", "t2", "t3", None, "t4"],
        },
        {
            "error_span_name": "db_connection_error",
            "count": 18,
            "percentage_of_errors": 25.7,
            "sample_trace_ids": ["t5", None, "t6"],
        },
        {
            "error_span_name": "validation_error",
            "count": 15,
            "percentage_of_errors": 21.4,
            "sample_trace_ids": ["t7", "t8", "t9", "t10"],
        },
        {
            "error_span_name": "auth_failure",
            "count": 8,
            "percentage_of_errors": 11.4,
            "sample_trace_ids": ["t11"],
        },
        {
            "error_span_name": "rate_limit_exceeded",
            "count": 4,
            "percentage_of_errors": 5.7,
            "sample_trace_ids": ["t12", "t13"],
        },
        # This 6th one should be ignored (only top 5)
        {
            "error_span_name": "misc_error",
            "count": 1,
            "percentage_of_errors": 0.1,
            "sample_trace_ids": ["t14"],
        },
    ]

    mock_store.execute_sql.side_effect = [
        [
            {
                "total_traces": 1000,
                "ok_count": 930,
                "error_count": 70,
                "error_rate_percentage": 7.0,
            }
        ],
        [
            {
                "p50_latency_ms": 150.0,
                "p90_latency_ms": 450.0,
                "p95_latency_ms": 650.0,
                "p99_latency_ms": 950.0,
                "max_latency_ms": 2500.0,
            }
        ],
        error_spans,  # Error spans query result
        [],  # No slow tools
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-01T12:00:00Z",
            }
        ],
        # Quality metrics
        [{"verbose_percentage": 5.0, "sample_trace_ids": []}],
        [{"rushed_complex_pct": 3.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 2.0, "sample_trace_ids": []}],
        [{"problematic_response_rate_percentage": 4.0, "sample_trace_ids": []}],
    ]

    census = mock_store.generate_census(experiment_id)

    # Verify only top 5 error spans are included
    assert len(census.operational_metrics.top_error_spans) == 5

    # Verify they're in correct order (by count)
    assert census.operational_metrics.top_error_spans[0].error_span_name == "api_call_timeout"
    assert census.operational_metrics.top_error_spans[0].count == 25
    # None values should be filtered out from sample_trace_ids
    assert census.operational_metrics.top_error_spans[0].sample_trace_ids == [
        "t1",
        "t2",
        "t3",
        "t4",
    ]

    assert census.operational_metrics.top_error_spans[4].error_span_name == "rate_limit_exceeded"
    assert census.operational_metrics.top_error_spans[4].count == 4

    # Verify "misc_error" is not included
    span_names = [span.error_span_name for span in census.operational_metrics.top_error_spans]
    assert "misc_error" not in span_names


def test_metadata_population(mock_store):
    """Test that census metadata is properly populated."""
    experiment_id = "12345"
    mock_store._get_trace_table_for_experiment.return_value = "custom.catalog.traces"

    # Mock minimal responses
    mock_store.execute_sql.side_effect = [
        [{"total_traces": 5, "ok_count": 5, "error_count": 0, "error_rate_percentage": 0.0}],
        [
            {
                "p50_latency_ms": 100.0,
                "p90_latency_ms": 100.0,
                "p95_latency_ms": 100.0,
                "p99_latency_ms": 100.0,
                "max_latency_ms": 100.0,
            }
        ],
        [],
        [],
        [
            {
                "first_trace_timestamp": "2024-01-01T00:00:00Z",
                "last_trace_timestamp": "2024-01-01T01:00:00Z",
            }
        ],
        [{"verbose_percentage": 0.0, "sample_trace_ids": []}],
        [{"rushed_complex_pct": 0.0, "sample_trace_ids": []}],
        [{"minimal_response_rate": 0.0, "sample_trace_ids": []}],
        [{"problematic_response_rate_percentage": 0.0, "sample_trace_ids": []}],
    ]

    census = mock_store.generate_census(experiment_id)

    # Verify metadata
    assert census.metadata.table_name == "custom.catalog.traces"
    assert isinstance(census.metadata.created_at, datetime)
    assert census.metadata.additional_metadata["experiment_id"] == experiment_id

    # Verify timestamp is set (don't check exact time due to timezone issues)
    # The datetime.now() in the code doesn't use timezone.utc
    assert census.metadata.created_at is not None
