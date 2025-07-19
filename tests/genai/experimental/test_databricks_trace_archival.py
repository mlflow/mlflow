"""
Tests for MLflow tracing databricks archival functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_archival import (
    SUPPORTED_SCHEMA_VERSION,
    _create_genai_trace_view,
    _validate_schema_versions,
    enable_databricks_trace_archival,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceLocation as ProtoTraceLocation,
)


def _create_mock_databricks_agents():
    """Helper function to create a mock databricks.agents module with proper __spec__."""
    mock_module = Mock()
    mock_module.__spec__ = Mock()
    mock_module.__spec__.name = "databricks.agents"
    return mock_module


def _create_trace_destination_proto(
    experiment_id: str = "12345",
    spans_table_name: str = "catalog.schema.spans",
    events_table_name: str = "catalog.schema.events",
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


@pytest.mark.parametrize(
    ("error_type", "expected_match"),
    [
        ("Connection timeout", "Failed to enable trace archival"),
        ("Authentication failed", "Failed to enable trace archival"),
        ("Network error", "Failed to enable trace archival"),
    ],
)
@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
def test_create_trace_destination_api_failures(mock_call_endpoint, error_type, expected_match):
    """Test various API failure scenarios."""
    mock_call_endpoint.side_effect = Exception(error_type)

    with patch("sys.modules", {"databricks.agents": Mock()}):
        with pytest.raises(MlflowException, match=expected_match):
            enable_databricks_trace_archival("12345", "catalog", "schema")


@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
def test_malformed_api_response(mock_call_endpoint):
    """Test handling of malformed API responses."""
    # Mock response missing required fields
    mock_response = Mock()
    mock_response.spans_table_name = "catalog.schema.spans"
    # Missing events_table_name intentionally
    mock_call_endpoint.return_value = mock_response

    with patch("sys.modules", {"databricks.agents": Mock()}):
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_trace_archival("12345", "catalog", "schema")


# Spark view creation tests


@patch("mlflow.genai.experimental.databricks_trace_archival._get_active_spark_session")
def test_create_genai_trace_view_with_active_session(mock_get_active_session):
    """Test creating GenAI trace view with an active Spark session."""
    # Mock active Spark session
    mock_spark = Mock()
    mock_get_active_session.return_value = mock_spark

    # Call the function directly
    _create_genai_trace_view(
        "catalog.schema.trace_logs_12345",
        "catalog.schema.spans_table",
        "catalog.schema.events_table",
    )

    # Verify SQL was executed
    mock_spark.sql.assert_called_once()
    sql_call = mock_spark.sql.call_args[0][0]
    assert "CREATE OR REPLACE VIEW catalog.schema.trace_logs_12345 AS" in sql_call
    assert "catalog.schema.spans_table" in sql_call
    assert "catalog.schema.events_table" in sql_call


@patch("mlflow.genai.experimental.databricks_trace_archival._get_active_spark_session")
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
        "catalog.schema.events_table",
    )

    # Verify fallback session was used
    mock_builder.getOrCreate.assert_called_once()
    mock_spark.sql.assert_called_once()


@patch("mlflow.genai.experimental.databricks_trace_archival._get_active_spark_session")
@patch("pyspark.sql.SparkSession.builder")
def test_create_genai_trace_view_spark_session_creation_fails(
    mock_builder, mock_get_active_session
):
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
            "catalog.schema.events_table",
        )


def test_databricks_agents_import_error():
    """Test that ImportError is raised when databricks-agents package is not available."""
    with pytest.raises(
        ImportError,
        match=r"The `mlflow\[databricks\]` package is required to use databricks trace archival",
    ):
        enable_databricks_trace_archival("12345", "catalog", "schema")


# Schema version validation tests


def test_validate_schema_versions_success():
    """Test successful schema version validation with supported version."""
    # Should not raise any exception
    _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION)


def test_unsupported_spans_schema_version():
    """Test that MlflowException is raised when spans table has unsupported schema version."""
    with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
        _validate_schema_versions("v2", SUPPORTED_SCHEMA_VERSION)


def test_unsupported_events_schema_version():
    """Test that MlflowException is raised when events table has unsupported schema version."""
    with pytest.raises(MlflowException, match="Unsupported events table schema version: v0"):
        _validate_schema_versions(SUPPORTED_SCHEMA_VERSION, "v0")


def test_both_unsupported_schema_versions():
    """Test that MlflowException is raised when both tables have unsupported schema versions."""
    # Should fail on spans version first
    with pytest.raises(MlflowException, match="Unsupported spans table schema version: invalid"):
        _validate_schema_versions("invalid", "also_invalid")


@patch("sys.modules", {"databricks.agents": Mock()})
@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival.get_databricks_host_creds")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_backend_returns_unsupported_spans_schema(
    mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint
):
    """Test end-to-end failure when backend returns unsupported spans schema version."""
    # Create proto response with unsupported spans schema version
    proto_response = _create_trace_destination_proto()
    proto_response.spans_schema_version = "v2"  # Unsupported version
    proto_response.events_schema_version = SUPPORTED_SCHEMA_VERSION
    mock_call_endpoint.return_value = proto_response

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
        enable_databricks_trace_archival("12345", "catalog", "schema")


@patch("sys.modules", {"databricks.agents": Mock()})
@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival.get_databricks_host_creds")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_backend_returns_unsupported_events_schema(
    mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint
):
    """Test end-to-end failure when backend returns unsupported events schema version."""
    # Create proto response with unsupported events schema version
    proto_response = _create_trace_destination_proto()
    proto_response.spans_schema_version = SUPPORTED_SCHEMA_VERSION
    proto_response.events_schema_version = "v0"  # Unsupported version
    mock_call_endpoint.return_value = proto_response

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with pytest.raises(MlflowException, match="Unsupported events table schema version: v0"):
        enable_databricks_trace_archival("12345", "catalog", "schema")


# Experiment tag setting tests


@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
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

    with patch("sys.modules", {"databricks.agents": Mock()}):
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_trace_archival("12345", "catalog", "schema")


@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival.get_databricks_host_creds")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_experiment_tag_setting(
    mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint
):
    """Test successful experiment tag setting."""

    proto_response = _create_trace_destination_proto()
    mock_call_endpoint.return_value = proto_response

    # Mock successful view creation
    mock_create_view.return_value = None

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("sys.modules", {"databricks.agents": Mock()}):
        result = enable_databricks_trace_archival("12345", "catalog", "schema")

    # Verify call_endpoint was called with correct arguments
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args

    # Validate call_endpoint arguments
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/trace-destinations"
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["host_creds"] == mock_get_creds.return_value

    # Validate JSON body contains properly serialized protobuf

    json_body = call_args.kwargs["json_body"]
    assert isinstance(json_body, str)  # Should be JSON string, not dict
    parsed_body = json.loads(json_body)  # Should parse without error
    assert parsed_body["uc_catalog"] == "catalog"
    assert parsed_body["uc_schema"] == "schema"
    assert parsed_body["uc_table_prefix"] == "trace_logs"

    # Validate set_experiment_tag was called with correct parameters
    from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE

    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,  # tag key
        "catalog.schema.trace_logs_12345",  # tag value (the archival view name)
    )
    assert result == "catalog.schema.trace_logs_12345"


# Successful archival integration tests


@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival.get_databricks_host_creds")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_archival_with_default_prefix(
    mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint
):
    """Test successful end-to-end archival with default table prefix."""

    proto_response = _create_trace_destination_proto(
        spans_table_name="catalog.schema.experiment_12345_spans",
        events_table_name="catalog.schema.experiment_12345_events",
    )
    mock_call_endpoint.return_value = proto_response

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("sys.modules", {"databricks.agents": Mock()}):
        result = enable_databricks_trace_archival("12345", "catalog", "schema")

    # Verify call_endpoint was called with correct arguments
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args

    # Validate arguments to call_endpoint
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/trace-destinations"
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["host_creds"] == mock_get_creds.return_value
    assert isinstance(call_args.kwargs["response_proto"], type(proto_response))

    # Validate JSON body contains properly serialized protobuf

    json_body = call_args.kwargs["json_body"]
    assert isinstance(json_body, str)  # Should be JSON string, not dict
    parsed_body = json.loads(json_body)  # Should parse without error

    # Validate protobuf fields in JSON
    assert parsed_body["uc_catalog"] == "catalog"
    assert parsed_body["uc_schema"] == "schema"
    assert parsed_body["uc_table_prefix"] == "trace_logs"
    assert parsed_body["trace_location"]["type"] == "MLFLOW_EXPERIMENT"
    assert parsed_body["trace_location"]["mlflow_experiment"]["experiment_id"] == "12345"
    mock_create_view.assert_called_once_with(
        "catalog.schema.trace_logs_12345",
        "catalog.schema.experiment_12345_spans",
        "catalog.schema.experiment_12345_events",
    )

    # Verify set_experiment_tag was called with correct parameters
    from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE

    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,  # tag key
        "catalog.schema.trace_logs_12345",  # tag value (archival location)
    )
    assert result == "catalog.schema.trace_logs_12345"


@patch("mlflow.genai.experimental.databricks_trace_archival.call_endpoint")
@patch("mlflow.genai.experimental.databricks_trace_archival.get_databricks_host_creds")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_successful_archival_with_custom_prefix(
    mock_mlflow_client, mock_create_view, mock_get_creds, mock_call_endpoint
):
    """Test successful archival with custom table prefix."""
    # Create proper protobuf response
    proto_response = _create_trace_destination_proto(
        spans_table_name="catalog.schema.custom_12345_spans",
        events_table_name="catalog.schema.custom_12345_events",
    )
    mock_call_endpoint.return_value = proto_response

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("sys.modules", {"databricks.agents": Mock()}):
        result = enable_databricks_trace_archival(
            "12345", "catalog", "schema", table_prefix="custom"
        )

    # Verify call_endpoint was called with correct arguments for custom prefix
    mock_call_endpoint.assert_called_once()
    call_args = mock_call_endpoint.call_args

    # Validate JSON body for custom prefix

    json_body = call_args.kwargs["json_body"]
    parsed_body = json.loads(json_body)
    assert parsed_body["uc_catalog"] == "catalog"
    assert parsed_body["uc_schema"] == "schema"
    assert parsed_body["trace_location"]["type"] == "MLFLOW_EXPERIMENT"
    assert parsed_body["trace_location"]["mlflow_experiment"]["experiment_id"] == "12345"
    assert parsed_body["uc_table_prefix"] == "custom"  # Should use custom prefix

    # Verify custom prefix was used
    mock_create_view.assert_called_once_with(
        "catalog.schema.custom_12345",
        "catalog.schema.custom_12345_spans",
        "catalog.schema.custom_12345_events",
    )

    # Verify set_experiment_tag was called with correct parameters for custom prefix
    from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE

    mock_client_instance.set_experiment_tag.assert_called_once_with(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,  # tag key
        "catalog.schema.custom_12345",  # tag value (custom archival location)
    )
    assert result == "catalog.schema.custom_12345"
