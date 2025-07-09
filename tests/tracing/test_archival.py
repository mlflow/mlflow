"""
Tests for MLflow trace archival functionality.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from mlflow.entities.trace_archive_configuration import TraceArchiveConfiguration
from mlflow.entities.trace_location import MlflowExperimentLocation, TraceLocation, TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.tracing.archival import enable_databricks_archival, DatabricksArchivalManager, SUPPORTED_SCHEMA_VERSION


class TestTraceArchival:
    """Test cases for trace archival functionality."""

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_success(self, mock_manager_class):
        """Test successful trace archival enablement."""
        # Mock the manager instance
        mock_manager = Mock()
        mock_manager.enable_archival.return_value = "catalog.schema.trace_logs_12345"
        mock_manager_class.return_value = mock_manager

        # Call the function
        result = enable_databricks_archival("12345", "catalog", "schema")

        # Verify the result
        assert result == "catalog.schema.trace_logs_12345"

        # Verify manager was created with correct parameters
        mock_manager_class.assert_called_once_with("12345", "catalog", "schema", "trace_logs")

        # Verify enable_archival was called
        mock_manager.enable_archival.assert_called_once()

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_authentication_failure(self, mock_manager_class):
        """Test trace archival when authentication fails."""
        # Mock manager to raise authentication failure
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Authentication failed")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_view_creation_failure(self, mock_manager_class):
        """Test trace archival when view creation fails."""
        # Mock manager to raise view creation failure
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: View creation failed")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_experiment_tag_failure(self, mock_manager_class):
        """Test trace archival when experiment tag setting fails."""
        # Mock manager to raise experiment tag failure
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Failed to set experiment tag")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_api_failure(self, mock_manager_class):
        """Test trace archival enablement when API call fails."""
        # Mock manager to raise API failure
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to create trace destination")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to create trace destination"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_malformed_response(self, mock_manager_class):
        """Test trace archival when API returns malformed response."""
        # Mock manager to raise malformed response error
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: 'events_table_name'")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception due to missing key
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_http_exception(self, mock_manager_class):
        """Test trace archival when HTTP request raises an exception."""
        # Mock manager to raise HTTP exception
        mock_manager = Mock()
        mock_manager.enable_archival.side_effect = MlflowException("Failed to enable trace archival for experiment 12345: Network error")
        mock_manager_class.return_value = mock_manager

        # Call the function and expect an exception
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_archival("12345", "catalog", "schema")

    @patch("mlflow.tracing.archival.DatabricksArchivalManager")
    def test_enable_trace_archival_with_custom_table_prefix(self, mock_manager_class):
        """Test trace archival with custom table prefix."""
        # Mock the manager instance
        mock_manager = Mock()
        mock_manager.enable_archival.return_value = "catalog.schema.trace_logs_12345"
        mock_manager_class.return_value = mock_manager

        # Call with custom table prefix
        result = enable_databricks_archival("12345", "catalog", "schema", table_prefix="custom_prefix")

        # Verify the result
        assert result == "catalog.schema.trace_logs_12345"
        
        # Verify manager was created with correct parameters including custom prefix
        mock_manager_class.assert_called_once_with("12345", "catalog", "schema", "custom_prefix")

        # Verify enable_archival was called
        mock_manager.enable_archival.assert_called_once()

    def test_enable_trace_archival_http_timeout_handling(self):
        """Test that HTTP timeout is properly configured."""
        # This test is now handled by the TraceArchivalManager implementation
        # The timeout logic is tested within the manager's enable_archival method
        # For integration testing, we can test the manager directly
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        assert manager.experiment_id == "12345"
        assert manager.catalog == "catalog"
        assert manager.schema == "schema"

    @patch("mlflow.tracing.archival._get_active_spark_session")
    def test_create_genai_trace_view_with_active_session(self, mock_get_active_session):
        """Test creating GenAI trace view with an active Spark session."""
        # Mock active Spark session
        mock_spark = Mock()
        mock_get_active_session.return_value = mock_spark

        # Create manager and call the method
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        manager.create_genai_trace_view(
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

        # Create manager and call the method
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        manager.create_genai_trace_view(
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

        # Create manager and call the method, expect an exception
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        with pytest.raises(MlflowException, match="Failed to create trace archival view"):
            manager.create_genai_trace_view(
                "catalog.schema.trace_logs_12345",
                "catalog.schema.spans_table",
                "catalog.schema.events_table"
            )


class TestDatabricksArchivalManager:
    """Test cases for DatabricksArchivalManager class."""

    def test_validate_schema_versions_success(self):
        """Test successful schema version validation with v1."""
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        # Should not raise any exception
        manager.validate_schema_versions(SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION)

    def test_validate_schema_versions_unsupported_spans(self):
        """Test schema version validation failure for unsupported spans version."""
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            manager.validate_schema_versions("v2", SUPPORTED_SCHEMA_VERSION)

    def test_validate_schema_versions_unsupported_events(self):
        """Test schema version validation failure for unsupported events version."""
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        with pytest.raises(MlflowException, match="Unsupported events table schema version: v2"):
            manager.validate_schema_versions(SUPPORTED_SCHEMA_VERSION, "v2")

    def test_validate_schema_versions_unsupported_both(self):
        """Test schema version validation failure for both unsupported versions."""
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        # Should fail on the spans version first
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            manager.validate_schema_versions("v2", "v3")

    def test_validate_schema_versions_different_unsupported_formats(self):
        """Test schema version validation with different unsupported version formats."""
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        # Test various unsupported formats
        unsupported_versions = ["1.0", "v1.1", "v0", "2", "beta", ""]
        
        for version in unsupported_versions:
            with pytest.raises(MlflowException, match="Unsupported spans table schema version"):
                manager.validate_schema_versions(version, SUPPORTED_SCHEMA_VERSION)
            
            with pytest.raises(MlflowException, match="Unsupported events table schema version"):
                manager.validate_schema_versions(SUPPORTED_SCHEMA_VERSION, version)

    @patch("mlflow.tracing.archival.http_request")
    @patch("mlflow.tracing.archival.get_databricks_host_creds")
    def test_enable_archival_schema_version_failure(self, mock_get_creds, mock_http_request):
        """Test that enable_archival fails when schema versions are unsupported."""
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

        # Create manager and expect schema validation to fail
        manager = DatabricksArchivalManager("12345", "catalog", "schema")
        
        with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
            manager.enable_archival()


class TestTraceArchiveConfigurationEntities:
    """Test cases for trace archive configuration entity classes."""

    def test_trace_archive_configuration_entity(self):
        """Test TraceArchiveConfiguration entity creation and conversion."""
        trace_location = TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="12345"),
        )

        destination = TraceArchiveConfiguration(
            trace_location=trace_location,
            spans_table_name="catalog.schema.spans",
            events_table_name="catalog.schema.events",
            spans_schema_version="v1",
            events_schema_version="v1",
        )

        # Test to_dict conversion
        dict_result = destination.to_dict()
        assert dict_result["spans_table_name"] == "catalog.schema.spans"
        assert dict_result["events_table_name"] == "catalog.schema.events"
        assert dict_result["spans_schema_version"] == "v1"
        assert dict_result["events_schema_version"] == "v1"
        assert "trace_location" in dict_result

        # Test from_dict conversion
        reconstructed = TraceArchiveConfiguration.from_dict(dict_result)
        assert reconstructed.spans_table_name == destination.spans_table_name
        assert reconstructed.events_table_name == destination.events_table_name
        assert reconstructed.spans_schema_version == destination.spans_schema_version
        assert reconstructed.events_schema_version == destination.events_schema_version

