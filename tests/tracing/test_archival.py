"""
Tests for MLflow trace archival functionality.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from mlflow.entities.trace_destination import TraceDestination
from mlflow.entities.trace_location import MlflowExperimentLocation, TraceLocation, TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.tracing.archival import enable_trace_archival, _create_genai_trace_view


class TestTraceArchival:
    """Test cases for trace archival functionality."""

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    @patch("mlflow.tracing.archival._create_genai_trace_view")
    @patch("mlflow.set_experiment_tag")
    def test_enable_trace_archival_success(
        self,
        mock_set_tag,
        mock_create_view,
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
        assert call_args[1]["endpoint"] == "/api/2.0/tracing/trace-destinations"
        assert call_args[1]["method"] == "POST"

        # Verify request body contains correct data
        request_json = call_args[1]["json"]
        assert request_json["trace_location"]["type"] == "MLFLOW_EXPERIMENT"
        assert request_json["trace_location"]["mlflow_experiment"]["experiment_id"] == "12345"
        assert request_json["uc_catalog"] == "catalog"
        assert request_json["uc_schema"] == "schema"
        assert request_json["uc_table_prefix"] == "trace_logs_12345"

        # Verify view creation was called (tables are created by backend now)
        mock_create_view.assert_called_once_with(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.experiment_12345_spans",
            "catalog.schema.experiment_12345_events",
        )

        # Verify experiment tag was set
        mock_set_tag.assert_called_once_with("trace_archival_table", "catalog.schema.trace_logs_12345")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_authentication_failure(self, mock_get_creds, mock_http_request):
        """Test trace archival when authentication fails."""
        # Mock authentication failure
        mock_get_creds.side_effect = Exception("Authentication failed")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_trace_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    @patch("mlflow.tracing.archival._create_genai_trace_view")
    def test_enable_trace_archival_view_creation_failure(self, mock_create_view, mock_get_creds, mock_http_request):
        """Test trace archival when view creation fails."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            "events_table_name": "catalog.schema.experiment_12345_events",
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Mock view creation failure
        mock_create_view.side_effect = Exception("View creation failed")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_trace_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    @patch("mlflow.tracing.archival._create_genai_trace_view")
    @patch("mlflow.set_experiment_tag")
    def test_enable_trace_archival_experiment_tag_failure(self, mock_set_tag, mock_create_view, mock_get_creds, mock_http_request):
        """Test trace archival when experiment tag setting fails."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            "events_table_name": "catalog.schema.experiment_12345_events",
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Mock experiment tag setting failure
        mock_set_tag.side_effect = Exception("Failed to set experiment tag")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_trace_archival("12345", "catalog", "schema")

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

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_malformed_response(self, mock_get_creds, mock_http_request):
        """Test trace archival when API returns malformed response."""
        # Mock HTTP response with missing fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            # Missing events_table_name
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Call the function and expect an exception due to missing key
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_trace_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_http_exception(self, mock_get_creds, mock_http_request):
        """Test trace archival when HTTP request raises an exception."""
        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Mock HTTP request to raise an exception
        mock_http_request.side_effect = Exception("Network error")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_trace_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_with_custom_table_prefix(self, mock_get_creds, mock_http_request):
        """Test trace archival with custom table prefix."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.custom_prefix_12345_spans",
            "events_table_name": "catalog.schema.custom_prefix_12345_events",
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        with patch("mlflow.tracing.archival._create_genai_trace_view") as mock_create_view, \
             patch("mlflow.set_experiment_tag") as mock_set_tag:
            
            # Call with custom table prefix
            result = enable_trace_archival("12345", "catalog", "schema", table_prefix="custom_prefix")

            # Verify view name always uses "trace_logs_" regardless of table_prefix
            # (table_prefix is only used for backend table creation)
            assert result == "catalog.schema.trace_logs_12345"
            
            # Verify request includes custom prefix
            call_args = mock_http_request.call_args
            request_json = call_args[1]["json"]
            assert request_json["uc_table_prefix"] == "custom_prefix_12345"

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_trace_archival_http_timeout_handling(self, mock_get_creds, mock_http_request):
        """Test that HTTP timeout is properly configured."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            "events_table_name": "catalog.schema.experiment_12345_events",
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        with patch("mlflow.tracing.archival._create_genai_trace_view"), \
             patch("mlflow.set_experiment_tag"), \
             patch("mlflow.tracing.archival.MLFLOW_HTTP_REQUEST_TIMEOUT") as mock_timeout:
            
            # Set a small timeout value
            mock_timeout.get.return_value = 5
            
            enable_trace_archival("12345", "catalog", "schema")

            # Verify timeout was used (minimum 10 seconds as per implementation)
            call_args = mock_http_request.call_args
            assert call_args[1]["timeout"] == 10  # max(5, 10) = 10

    @patch("mlflow.tracing.archival._get_active_spark_session")
    def test_create_genai_trace_view_with_active_session(self, mock_get_active_session):
        """Test creating GenAI trace view with an active Spark session."""
        # Mock active Spark session
        mock_spark = Mock()
        mock_get_active_session.return_value = mock_spark

        # Call the function
        _create_genai_trace_view(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.spans_table",
            "catalog.schema.events_table"
        )

        # Verify SQL was executed
        mock_spark.sql.assert_called_once()
        sql_call = mock_spark.sql.call_args[0][0]
        assert "CREATE OR REPLACE VIEW catalog.schema.trace_logs_12345 AS" in sql_call
        assert "catalog.schema.spans_table" in sql_call
        assert "catalog.schema.events_table" in sql_call

    @patch("mlflow.tracing.archival._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_genai_trace_view_with_fallback_session(self, mock_builder, mock_get_active_session):
        """Test creating view when no active session but can create a new one."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession builder
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark

        # Call the function
        _create_genai_trace_view(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.spans_table",
            "catalog.schema.events_table"
        )

        # Verify fallback session was used
        mock_builder.getOrCreate.assert_called_once()
        mock_spark.sql.assert_called_once()

    @patch("mlflow.tracing.archival._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_genai_trace_view_spark_session_creation_fails(self, mock_builder, mock_get_active_session):
        """Test creating view when Spark session creation fails."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession.builder.getOrCreate to raise an exception
        mock_builder.getOrCreate.side_effect = Exception("Failed to create Spark session")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to create trace archival view"):
            _create_genai_trace_view(
                "catalog.schema.trace_logs_12345",
                "catalog.schema.spans_table",
                "catalog.schema.events_table"
            )


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

