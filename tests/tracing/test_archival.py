"""
Tests for MLflow trace archival functionality.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from mlflow.entities.trace_destination import TraceDestination
from mlflow.entities.trace_location import MlflowExperimentLocation, TraceLocation, TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.tracing.archival import enable_trace_archival, _create_spans_table, _create_events_table


class TestTraceArchival:
    """Test cases for trace archival functionality."""

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    @patch("mlflow.tracing.archival.create_spans_table")
    @patch("mlflow.tracing.archival.create_events_table")
    @patch("mlflow.tracing.archival.create_genai_trace_view")
    @patch("mlflow.set_experiment_tag")
    def test_enable_trace_archival_success(
        self,
        mock_set_tag,
        mock_create_view,
        mock_create_events,
        mock_create_spans,
        mock_get_creds,
        mock_http_request,
    ):
        """Test successful trace archival enablement."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "trace_location": {
                "type": "mlflow_experiment",
                "mlflow_experiment": {"experiment_id": "12345"},
            },
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            "events_table_name": "catalog.schema.experiment_12345_events",
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Call the function
        result = enable_trace_archival("12345", "catalog", "schema")

        # Verify the result
        assert result == "catalog.schema.trace_logs_12345"

        # Verify HTTP request was made correctly
        mock_http_request.assert_called_once()
        call_args = mock_http_request.call_args
        assert call_args[1]["endpoint"] == "/2.0/tracing/trace-destinations"
        assert call_args[1]["method"] == "POST"

        # Verify table creation was called
        mock_create_spans.assert_called_once_with("catalog.schema.experiment_12345_spans")
        mock_create_events.assert_called_once_with("catalog.schema.experiment_12345_events")

        # Verify view creation was called
        mock_create_view.assert_called_once_with(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.experiment_12345_spans",
            "catalog.schema.experiment_12345_events",
        )

        # Verify experiment tag was set
        mock_set_tag.assert_called_once_with("trace_archival_table", "catalog.schema.trace_logs_12345")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_api_failure(self, mock_get_creds, mock_http_request):
        """Test trace archival enablement when API call fails."""
        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to create trace destination"):
            enable_trace_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival._get_active_spark_session")
    def test_create_spans_table_with_active_session(self, mock_get_active_session):
        """Test creating spans table with an active Spark session."""
        # Mock active Spark session
        mock_spark = Mock()
        mock_get_active_session.return_value = mock_spark

        # Call the function
        _create_spans_table("catalog.schema.spans_table")

        # Verify SQL was executed
        mock_spark.sql.assert_called_once()
        sql_call = mock_spark.sql.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS catalog.schema.spans_table" in sql_call
        assert "trace_id STRING" in sql_call
        assert "USING DELTA" in sql_call

    @patch("mlflow.tracing.archival._get_active_spark_session")
    def test_create_events_table_with_active_session(self, mock_get_active_session):
        """Test creating events table with an active Spark session."""
        # Mock active Spark session
        mock_spark = Mock()
        mock_get_active_session.return_value = mock_spark

        # Call the function
        _create_events_table("catalog.schema.events_table")

        # Verify SQL was executed
        mock_spark.sql.assert_called_once()
        sql_call = mock_spark.sql.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS catalog.schema.events_table" in sql_call
        assert "event_name STRING" in sql_call
        assert "USING DELTA" in sql_call

    @patch("mlflow.tracing.archival._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_table_with_fallback_session(self, mock_builder, mock_get_active_session):
        """Test creating table when no active session but can create a new one."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession builder
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark

        # Call the function
        _create_spans_table("catalog.schema.spans_table")

        # Verify fallback session was used
        mock_builder.getOrCreate.assert_called_once()
        mock_spark.sql.assert_called_once()

    @patch("mlflow.tracing.archival._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_table_spark_session_creation_fails(self, mock_builder, mock_get_active_session):
        """Test creating table when Spark session creation fails."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession.builder.getOrCreate to raise an exception
        mock_builder.getOrCreate.side_effect = Exception("Failed to create Spark session")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to create spans table"):
            _create_spans_table("catalog.schema.spans_table")


class TestTraceDestinationEntities:
    """Test cases for trace destination entity classes."""

    def test_trace_destination_entity(self):
        """Test TraceDestination entity creation and conversion."""
        trace_location = TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="12345"),
        )

        destination = TraceDestination(
            trace_location=trace_location,
            spans_table_name="catalog.schema.spans",
            events_table_name="catalog.schema.events",
        )

        # Test to_dict conversion
        dict_result = destination.to_dict()
        assert dict_result["spans_table_name"] == "catalog.schema.spans"
        assert dict_result["events_table_name"] == "catalog.schema.events"
        assert "trace_location" in dict_result

        # Test from_dict conversion
        reconstructed = TraceDestination.from_dict(dict_result)
        assert reconstructed.spans_table_name == destination.spans_table_name
        assert reconstructed.events_table_name == destination.events_table_name

