"""
Tests for MLflow tracing databricks archival functionality.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from mlflow.entities.trace_archive_configuration import TraceArchiveConfiguration
from mlflow.entities.trace_location import MlflowExperimentLocation, TraceLocation, TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.tracing.archival.databricks import enable_databricks_archival, _validate_schema_versions, _create_genai_trace_view, _do_enable_databricks_archival, SUPPORTED_SCHEMA_VERSION


class TestTraceArchival:
    """Test cases for trace archival functionality."""

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_success(self, mock_impl):
        """Test successful trace archival enablement."""
        # Mock the implementation function
        mock_impl.return_value = "catalog.schema.trace_logs_12345"

        # Call the function
        result = enable_databricks_archival("12345", "catalog", "schema")

        # Verify the result
        assert result == "catalog.schema.trace_logs_12345"

        # Verify implementation was called with correct parameters
        mock_impl.assert_called_once_with("12345", "catalog", "schema", "trace_logs")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_authentication_failure(self, mock_impl):
        """Test trace archival when authentication fails."""
        # Mock implementation to raise authentication failure
        mock_impl.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Authentication failed")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_view_creation_failure(self, mock_impl):
        """Test trace archival when view creation fails."""
        # Mock implementation to raise view creation failure
        mock_impl.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: View creation failed")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_experiment_tag_failure(self, mock_impl):
        """Test trace archival when experiment tag setting fails."""
        # Mock implementation to raise experiment tag failure
        mock_impl.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Failed to set experiment tag")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_api_failure(self, mock_impl):
        """Test trace archival enablement when API call fails."""
        # Mock implementation to raise API failure
        mock_impl.side_effect = MlflowException("Failed to create trace destination")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to create trace destination"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_malformed_response(self, mock_impl):
        """Test trace archival when API returns malformed response."""
        # Mock implementation to raise malformed response error
        mock_impl.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: 'events_table_name'")

        # Call the function and expect an exception due to missing key
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_http_exception(self, mock_impl):
        """Test trace archival when HTTP request raises an exception."""
        # Mock implementation to raise HTTP exception
        mock_impl.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Network error")

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._do_enable_databricks_archival")
    def test_enable_trace_archival_with_custom_table_prefix(self, mock_impl):
        """Test trace archival with custom table prefix."""
        # Mock the implementation function
        mock_impl.return_value = "catalog.schema.trace_logs_12345"

        # Call with custom table prefix
        result = enable_databricks_archival("12345", "catalog", "schema", table_prefix="custom_prefix")

        # Verify the result
        assert result == "catalog.schema.trace_logs_12345"
        
        # Verify implementation was called with correct parameters including custom prefix
        mock_impl.assert_called_once_with("12345", "catalog", "schema", "custom_prefix")

    def test_enable_trace_archival_http_timeout_handling(self):
        """Test that HTTP timeout is properly configured."""
        # This test is now handled by the implementation function
        # The timeout logic is tested within the _enable_databricks_archival_impl function
        # For integration testing, we can test the timeout by mocking the HTTP request
        with patch("mlflow.tracing.archival.databricks.http_request") as mock_http_request:
            mock_http_request.side_effect = Exception("Connection timeout")
            
            with pytest.raises(MlflowException, match="Failed to enable trace archival"):
                enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.databricks._get_active_spark_session")
    def test_create_genai_trace_view_with_active_session(self, mock_get_active_session):
        """Test creating GenAI trace view with an active Spark session."""
        # Mock active Spark session
        mock_spark = Mock()
        mock_get_active_session.return_value = mock_spark

        # Call the function directly
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

    @patch("mlflow.tracing.archival.databricks._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_genai_trace_view_with_fallback_session(self, mock_builder, mock_get_active_session):
        """Test creating view when no active session but can create a new one."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession builder
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark

        # Call the function directly
        _create_genai_trace_view(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.spans_table",
            "catalog.schema.events_table"
        )

        # Verify fallback session was used
        mock_builder.getOrCreate.assert_called_once()
        mock_spark.sql.assert_called_once()

    @patch("mlflow.tracing.archival.databricks._get_active_spark_session")
    @patch("pyspark.sql.SparkSession.builder")
    def test_create_genai_trace_view_spark_session_creation_fails(self, mock_builder, mock_get_active_session):
        """Test creating view when Spark session creation fails."""
        # Mock no active session
        mock_get_active_session.return_value = None

        # Mock SparkSession.builder.getOrCreate to raise an exception
        mock_builder.getOrCreate.side_effect = Exception("Failed to create Spark session")

        # Call the function directly and expect an exception
        with pytest.raises(MlflowException, match="Failed to create trace archival view"):
            _create_genai_trace_view(
                "catalog.schema.trace_logs_12345",
                "catalog.schema.spans_table",
                "catalog.schema.events_table"
            )


class TestDatabricksArchivalFunctions:
    """Test cases for individual Databricks archival functions."""

    def test_validate_schema_versions_success(self):
        """Test successful schema version validation with v1."""
        # Should not raise any exception
        _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION)

    def test_validate_schema_versions_unsupported_spans(self):
        """Test schema version validation failure for unsupported spans version."""
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            _validate_schema_versions("v2", SUPPORTED_SCHEMA_VERSION)

    def test_validate_schema_versions_unsupported_events(self):
        """Test schema version validation failure for unsupported events version."""
        with pytest.raises(MlflowException, match="Unsupported events table schema version: v2"):
            _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, "v2")

    def test_validate_schema_versions_unsupported_both(self):
        """Test schema version validation failure for both unsupported versions."""
        # Should fail on the spans version first
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            _validate_schema_versions("v2", "v3")

    def test_validate_schema_versions_different_unsupported_formats(self):
        """Test schema version validation with different unsupported version formats."""
        # Test various unsupported formats
        unsupported_versions = ["1.0", "v1.1", "v0", "2", "beta", ""]
        
        for version in unsupported_versions:
            with pytest.raises(MlflowException, match="Unsupported spans table schema version"):
                _validate_schema_versions(version, SUPPORTED_SCHEMA_VERSION)
            
            with pytest.raises(MlflowException, match="Unsupported events table schema version"):
                _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, version)

    @patch("mlflow.tracing.archival.databricks.http_request")
    @patch("mlflow.tracing.archival.databricks.get_databricks_host_creds")
    def test_enable_archival_schema_version_failure(self, mock_get_creds, mock_http_request):
        """Test that _enable_databricks_archival_impl fails when schema versions are unsupported."""
        # Mock successful HTTP response with unsupported schema version
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "spans_table_name": "catalog.schema.experiment_12345_spans",
            "events_table_name": "catalog.schema.experiment_12345_events",
            "spans_schema_version": "v2",  # Unsupported version
            "events_schema_version": SUPPORTED_SCHEMA_VERSION,
        }
        mock_http_request.return_value = mock_response

        # Mock credentials
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds

        # Call the implementation directly and expect schema validation to fail
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            _do_enable_databricks_archival("12345", "catalog", "schema")


class TestTraceArchiveConfigurationEntities:
    """Test cases for trace archive configuration entity classes."""

    def test_trace_archive_configuration_entity(self):
        """Test TraceArchiveConfiguration entity creation and conversion."""
        destination = TraceArchiveConfiguration(
            experiment_id="12345",
            spans_table_name="catalog.schema.spans",
            events_table_name="catalog.schema.events",
            spans_schema_version="v1",
            events_schema_version="v1",
        )

        # Test to_dict conversion
        dict_result = destination.to_dict()
        assert dict_result["experiment_id"] == "12345"
        assert dict_result["spans_table_name"] == "catalog.schema.spans"
        assert dict_result["events_table_name"] == "catalog.schema.events"
        assert dict_result["spans_schema_version"] == "v1"
        assert dict_result["events_schema_version"] == "v1"

        # Test from_dict conversion
        reconstructed = TraceArchiveConfiguration.from_dict(dict_result)
        assert reconstructed.experiment_id == destination.experiment_id
        assert reconstructed.spans_table_name == destination.spans_table_name
        assert reconstructed.events_table_name == destination.events_table_name
        assert reconstructed.spans_schema_version == destination.spans_schema_version
        assert reconstructed.events_schema_version == destination.events_schema_version

    def test_trace_archive_configuration_from_proto_validation(self):
        """Test that from_proto validates only experiment locations are supported."""
        from mlflow.protos.databricks_trace_server_pb2 import TraceDestination as ProtoTraceDestination
        from mlflow.protos.databricks_trace_server_pb2 import TraceLocation as ProtoTraceLocation
        
        # Test with valid experiment location
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = "12345"
        
        proto_destination = ProtoTraceDestination()
        proto_destination.trace_location.CopyFrom(proto_trace_location)
        proto_destination.spans_table_name = "catalog.schema.spans"
        proto_destination.events_table_name = "catalog.schema.events"
        proto_destination.spans_schema_version = "v1"
        proto_destination.events_schema_version = "v1"
        
        # This should work fine
        config = TraceArchiveConfiguration.from_proto(proto_destination)
        assert config.experiment_id == "12345"
        
        # Test with invalid inference table location
        proto_trace_location_invalid = ProtoTraceLocation()
        proto_trace_location_invalid.type = ProtoTraceLocation.TraceLocationType.INFERENCE_TABLE
        proto_trace_location_invalid.inference_table.full_table_name = "catalog.schema.table"
        
        proto_destination_invalid = ProtoTraceDestination()
        proto_destination_invalid.trace_location.CopyFrom(proto_trace_location_invalid)
        proto_destination_invalid.spans_table_name = "catalog.schema.spans"
        proto_destination_invalid.events_table_name = "catalog.schema.events"
        proto_destination_invalid.spans_schema_version = "v1"
        proto_destination_invalid.events_schema_version = "v1"
        
        # This should raise an exception
        with pytest.raises(MlflowException, match="TraceArchiveConfiguration only supports MLflow experiments"):
            TraceArchiveConfiguration.from_proto(proto_destination_invalid)

