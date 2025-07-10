"""
Tests for MLflow tracing databricks archival functionality.
"""

import pytest
from unittest.mock import Mock, patch

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
    TraceLocation as ProtoTraceLocation,
)
from mlflow.tracing.archival.databricks import (
    enable_databricks_archival, 
    _validate_schema_versions, 
    _create_genai_trace_view, 
    SUPPORTED_SCHEMA_VERSION
)


def _create_trace_destination_proto(
    experiment_id: str = "12345",
    spans_table_name: str = "catalog.schema.spans", 
    events_table_name: str = "catalog.schema.events"
) -> ProtoTraceDestination:
    """Helper function to create a ProtoTraceDestination object for testing."""
    proto_trace_location = ProtoTraceLocation()
    proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    proto_trace_location.mlflow_experiment.experiment_id = experiment_id
    
    proto_response = ProtoTraceDestination()
    proto_response.trace_location.CopyFrom(proto_trace_location)
    proto_response.spans_table_name = spans_table_name
    proto_response.events_table_name = events_table_name
    proto_response.spans_schema_version = SUPPORTED_SCHEMA_VERSION
    proto_response.events_schema_version = SUPPORTED_SCHEMA_VERSION
    
    return proto_response


# CreateTraceDestination API failure tests

@pytest.mark.parametrize("error_type,expected_match", [
    ("Connection timeout", "Failed to enable Databricks archival"),
    ("Authentication failed", "Failed to enable Databricks archival"),
    ("Network error", "Failed to enable Databricks archival"),
])
@patch("mlflow.tracing.archival.databricks.call_endpoint")
def test_create_trace_destination_api_failures(mock_call_endpoint, error_type, expected_match):
    """Test various API failure scenarios."""
    mock_call_endpoint.side_effect = Exception(error_type)
    
    with pytest.raises(MlflowException, match=expected_match):
        enable_databricks_archival("12345", "catalog", "schema")


@patch("mlflow.tracing.archival.databricks.call_endpoint")
def test_malformed_api_response(mock_call_endpoint):
    """Test handling of malformed API responses."""
    # Mock response missing required fields
    mock_response = Mock()
    mock_response.spans_table_name = "catalog.schema.spans"
    # Missing events_table_name intentionally
    mock_call_endpoint.return_value = mock_response
    
    with pytest.raises(MlflowException, match="Failed to enable Databricks archival"):
        enable_databricks_archival("12345", "catalog", "schema")


# Spark view creation tests

@patch("mlflow.tracing.archival.databricks._get_active_spark_session")
def test_create_genai_trace_view_with_active_session(mock_get_active_session):
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
def test_create_genai_trace_view_with_fallback_session(mock_builder, mock_get_active_session):
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
def test_create_genai_trace_view_spark_session_creation_fails(mock_builder, mock_get_active_session):
    """Test creating view when Spark session creation fails."""
    # Mock no active session
    mock_get_active_session.return_value = None

    # Mock SparkSession.builder.getOrCreate to raise an exception
    mock_builder.getOrCreate.side_effect = Exception("Failed to create Spark session")

    # Call the function directly and expect an exception
    with pytest.raises(MlflowException, match="Failed to configure Databricks trace archival"):
        _create_genai_trace_view(
            "catalog.schema.trace_logs_12345",
            "catalog.schema.spans_table",
            "catalog.schema.events_table"
        )


# Schema version validation tests

def test_validate_schema_versions_success():
    """Test successful schema version validation with v1."""
    # Should not raise any exception
    _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION)


def test_validate_schema_versions_unsupported():
    """Test schema version validation failure for unsupported version."""
    with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
        _validate_schema_versions("v2", SUPPORTED_SCHEMA_VERSION)



# Experiment tag setting tests

@patch("mlflow.tracing.archival.databricks.call_endpoint")
@patch("mlflow.tracing.archival.databricks._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_experiment_tag_setting_failure(mock_mlflow_client, mock_create_view, mock_call_endpoint):
    """Test experiment tag setting failure."""

    proto_response = _create_trace_destination_proto()
    mock_call_endpoint.return_value = proto_response
    
    # Mock view creation to succeed
    mock_create_view.return_value = None
    
    # Mock client to raise exception on set_experiment_tag
    mock_client_instance = Mock()
    mock_client_instance.set_experiment_tag.side_effect = Exception("Permission denied")
    mock_mlflow_client.return_value = mock_client_instance
    
    with pytest.raises(MlflowException, match="Failed to enable Databricks archival"):
        enable_databricks_archival("12345", "catalog", "schema")


@patch("mlflow.tracing.archival.databricks.call_endpoint")
@patch("mlflow.tracing.archival.databricks.get_databricks_host_creds")
@patch("mlflow.tracing.archival.databricks._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_experiment_tag_setting(mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint):
    """Test successful experiment tag setting."""

    proto_response = _create_trace_destination_proto()
    mock_call_endpoint.return_value = proto_response
    
    # Mock successful view creation
    mock_create_view.return_value = None
    
    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance
    
    result = enable_databricks_archival("12345", "catalog", "schema")
    
    # Verify call_endpoint was called with correct arguments
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args
    
    # Validate call_endpoint arguments
    assert call_args.kwargs['endpoint'] == "/api/2.0/tracing/trace-destinations"
    assert call_args.kwargs['method'] == "POST"
    assert call_args.kwargs['host_creds'] == mock_get_creds.return_value
    
    # Validate JSON body contains properly serialized protobuf
    import json
    json_body = call_args.kwargs['json_body']
    assert isinstance(json_body, str)  # Should be JSON string, not dict
    parsed_body = json.loads(json_body)  # Should parse without error
    assert parsed_body['uc_catalog'] == "catalog"
    assert parsed_body['uc_schema'] == "schema"
    assert parsed_body['uc_table_prefix'] == "trace_logs"
    
    # Validate set_experiment_tag was called with correct parameters
    from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE
    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE,  # tag key
        "catalog.schema.trace_logs_12345"  # tag value (the archival view name)
    )
    assert result == "catalog.schema.trace_logs_12345"


# Successful archival integration tests

@patch("mlflow.tracing.archival.databricks.call_endpoint")
@patch("mlflow.tracing.archival.databricks.get_databricks_host_creds")
@patch("mlflow.tracing.archival.databricks._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_archival_with_default_prefix(mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint):
    """Test successful end-to-end archival with default table prefix."""
    
    proto_response = _create_trace_destination_proto(
        spans_table_name="catalog.schema.experiment_12345_spans",
        events_table_name="catalog.schema.experiment_12345_events"
    )
    mock_call_endpoint.return_value = proto_response
    
    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance
    
    result = enable_databricks_archival("12345", "catalog", "schema")
    
    # Verify call_endpoint was called with correct arguments
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args
    
    # Validate arguments to call_endpoint
    assert call_args.kwargs['endpoint'] == "/api/2.0/tracing/trace-destinations"
    assert call_args.kwargs['method'] == "POST"
    assert call_args.kwargs['host_creds'] == mock_get_creds.return_value
    assert isinstance(call_args.kwargs['response_proto'], type(proto_response))
    
    # Validate JSON body contains properly serialized protobuf
    import json
    json_body = call_args.kwargs['json_body']
    assert isinstance(json_body, str)  # Should be JSON string, not dict
    parsed_body = json.loads(json_body)  # Should parse without error
    
    # Validate protobuf fields in JSON
    assert parsed_body['uc_catalog'] == "catalog"
    assert parsed_body['uc_schema'] == "schema" 
    assert parsed_body['uc_table_prefix'] == "trace_logs"
    assert parsed_body['trace_location']['type'] == "MLFLOW_EXPERIMENT"
    assert parsed_body['trace_location']['mlflow_experiment']['experiment_id'] == "12345"
    mock_create_view.assert_called_once_with(
        "catalog.schema.trace_logs_12345",
        "catalog.schema.experiment_12345_spans",
        "catalog.schema.experiment_12345_events"
    )
    
    # Verify set_experiment_tag was called with correct parameters
    from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE
    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE,  # tag key
        "catalog.schema.trace_logs_12345"  # tag value (archival location)
    )
    assert result == "catalog.schema.trace_logs_12345"


@patch("mlflow.tracing.archival.databricks.call_endpoint")
@patch("mlflow.tracing.archival.databricks.get_databricks_host_creds")
@patch("mlflow.tracing.archival.databricks._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_archival_with_custom_prefix(mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint):
    """Test successful archival with custom table prefix."""
    # Create proper protobuf response  
    proto_response = _create_trace_destination_proto(
        spans_table_name="catalog.schema.custom_12345_spans",
        events_table_name="catalog.schema.custom_12345_events"
    )
    mock_call_endpoint.return_value = proto_response
    
    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance
    
    result = enable_databricks_archival("12345", "catalog", "schema", table_prefix="custom")
    
    # Verify call_endpoint was called with correct arguments for custom prefix
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args
    
    # Validate JSON body for custom prefix
    import json
    json_body = call_args.kwargs['json_body']
    parsed_body = json.loads(json_body)
    assert parsed_body['uc_catalog'] == "catalog"
    assert parsed_body['uc_schema'] == "schema" 
    assert parsed_body['trace_location']['type'] == "MLFLOW_EXPERIMENT"
    assert parsed_body['trace_location']['mlflow_experiment']['experiment_id'] == "12345"
    assert parsed_body['uc_table_prefix'] == "custom"  # Should use custom prefix
    
    # Verify custom prefix was used
    mock_create_view.assert_called_once_with(
        "catalog.schema.custom_12345",
        "catalog.schema.custom_12345_spans",
        "catalog.schema.custom_12345_events"
    )
    
    # Verify set_experiment_tag was called with correct parameters for custom prefix
    from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE
    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE,  # tag key
        "catalog.schema.custom_12345"  # tag value (custom archival location)
    )
    assert result == "catalog.schema.custom_12345"
