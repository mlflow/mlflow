"""
Tests for MLflow tracing databricks archival functionality.
"""

from unittest.mock import Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_archival import (
    SUPPORTED_SCHEMA_VERSION,
    _create_genai_trace_view,
    _validate_schema_versions,
    enable_databricks_trace_archival,
    set_experiment_storage_location,
)
from mlflow.genai.experimental.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceLocation as ProtoTraceLocation,
)
from mlflow.tracing.destination import DatabricksUnityCatalog
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED,
    MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,
)


@pytest.fixture
def mock_workspace_id():
    """Mock workspace ID for testing."""
    return "123"


@pytest.fixture
def mock_workspace_client(mock_workspace_id):
    """Mock WorkspaceClient for testing."""
    client = Mock()
    client.get_workspace_id.return_value = mock_workspace_id
    return client


def _create_mock_databricks_agents():
    """Helper function to create a mock databricks.agents module with proper __spec__."""
    mock_module = Mock()
    mock_module.__spec__ = Mock()
    mock_module.__spec__.name = "databricks.agents"
    return mock_module


@pytest.fixture
def mock_delta_archival_mixin():
    """Fixture that provides a properly mocked DatabricksDeltaArchivalMixin with cache setup."""
    with patch(
        "mlflow.genai.experimental.databricks_trace_archival.DatabricksDeltaArchivalMixin"
    ) as mock_mixin:
        # Setup cache mocking for testing cache clearing behavior
        mock_mixin._config_cache_lock = Mock()
        mock_mixin._config_cache_lock.__enter__ = Mock()
        mock_mixin._config_cache_lock.__exit__ = Mock()
        mock_mixin._config_cache = {}  # Start with empty cache, tests can populate as needed
        yield mock_mixin


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
@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_create_trace_destination_api_failures(
    mock_mlflow_client,
    mock_trace_client,
    mock_workspace_client_class,
    error_type,
    expected_match,
    mock_workspace_client,
):
    """Test various API failure scenarios."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Mock trace client to raise exception
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.side_effect = Exception(error_type)
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful MLflow client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(MlflowException, match=expected_match):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_malformed_api_response(
    mock_mlflow_client, mock_trace_client, mock_workspace_client_class, mock_workspace_client
):
    """Test handling of malformed API responses."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Mock trace client to return malformed config (missing events_table_name)
    mock_config = Mock()
    mock_config.spans_table_name = "catalog.schema.spans"
    # Missing events_table_name intentionally
    del mock_config.events_table_name  # Make sure it doesn't have this attribute

    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful MLflow client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")


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


@patch("importlib.util.find_spec", return_value=Mock())
@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_backend_returns_unsupported_spans_schema(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_find_spec,
    mock_workspace_client,
):
    """Test end-to-end failure when backend returns unsupported spans schema version."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create config with unsupported spans schema version
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        events_table_name="catalog.schema.events",
        spans_schema_version="v2",  # Unsupported version
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful MLflow client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with pytest.raises(MlflowException, match="Unsupported spans table schema version: v2"):
        enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")


@patch("importlib.util.find_spec", return_value=Mock())
@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_backend_returns_unsupported_events_schema(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_find_spec,
    mock_workspace_client,
):
    """Test end-to-end failure when backend returns unsupported events schema version."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create config with unsupported events schema version
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        events_table_name="catalog.schema.events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version="v0",  # Unsupported version
    )

    # Mock trace client to return config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful MLflow client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with pytest.raises(MlflowException, match="Unsupported events table schema version: v0"):
        enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")


# Experiment tag setting tests


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.tracking.MlflowClient")
def test_experiment_tag_setting_failure(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test experiment tag setting failure."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create a valid config
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        events_table_name="catalog.schema.events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return valid config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock view creation to succeed
    mock_create_view.return_value = None

    # Mock client to raise exception on set_experiment_tag
    mock_client_instance = Mock()
    mock_client_instance.set_experiment_tag.side_effect = Exception("Permission denied")
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(MlflowException, match="Failed to enable trace archival"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_successful_experiment_tag_setting(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test successful experiment tag setting."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create a valid config
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        events_table_name="catalog.schema.events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return valid config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful view creation
    mock_create_view.return_value = None

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_experiment = Mock()
    mock_experiment.tags = {}  # No existing archival tag
    mock_client_instance.get_experiment.return_value = mock_experiment
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        result = enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify trace client was called with correct arguments
    mock_trace_client_instance.create_trace_destination.assert_called_once_with(
        experiment_id="12345",
        catalog="catalog",
        schema="schema",
        table_prefix="prefix",
    )

    # Validate set_experiment_tag was called twice with correct parameters
    assert mock_client_instance.set_experiment_tag.call_count == 2

    # Verify storage table tag
    mock_client_instance.set_experiment_tag.assert_any_call(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,  # tag key
        "catalog.schema.prefix_experiment_123_12345_genai_view",  # tag value (the view name)
    )

    # Verify rolling deletion tag
    mock_client_instance.set_experiment_tag.assert_any_call(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED,  # tag key
        "true",  # tag value
    )
    assert result == "catalog.schema.prefix_experiment_123_12345_genai_view"


# Successful archival integration tests


@pytest.mark.parametrize(
    ("table_prefix", "expected_view_name", "expected_spans_table", "expected_events_table"),
    [
        (
            "custom",  # custom prefix
            "catalog.schema.custom_experiment_123_12345_genai_view",
            "catalog.schema.custom_12345_spans",
            "catalog.schema.custom_12345_events",
        ),
    ],
)
@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_successful_archival_with_prefix(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    table_prefix,
    expected_view_name,
    expected_spans_table,
    expected_events_table,
    mock_workspace_client,
):
    """Test successful end-to-end archival with different table prefixes."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create a valid config
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name=expected_spans_table,
        events_table_name=expected_events_table,
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return valid config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_experiment = Mock()
    mock_experiment.tags = {}  # No existing archival tag
    mock_client_instance.get_experiment.return_value = mock_experiment
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        result = enable_databricks_trace_archival(
            "12345", "catalog", "schema", table_prefix=table_prefix
        )

    # Verify trace client was called with correct arguments
    mock_trace_client_instance.create_trace_destination.assert_called_once_with(
        experiment_id="12345",
        catalog="catalog",
        schema="schema",
        table_prefix=table_prefix,
    )

    mock_create_view.assert_called_once_with(
        expected_view_name,
        expected_spans_table,
        expected_events_table,
    )

    # Verify set_experiment_tag was called twice with correct parameters
    assert mock_client_instance.set_experiment_tag.call_count == 2

    # Verify storage table tag
    mock_client_instance.set_experiment_tag.assert_any_call(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,  # tag key
        expected_view_name,  # tag value (archival location)
    )

    # Verify rolling deletion tag
    mock_client_instance.set_experiment_tag.assert_any_call(
        "12345",  # experiment_id
        MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED,  # tag key
        "true",  # tag value
    )
    assert result == expected_view_name


# Idempotency tests


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_idempotent_enablement(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test that enable_databricks_trace_archival is idempotent when called multiple times."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.experiment_12345_spans",
        events_table_name="catalog.schema.experiment_12345_events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return valid config (simulating existing archival)
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_experiment = Mock()
    mock_experiment.tags = {}  # No existing archival tag
    mock_client_instance.get_experiment.return_value = mock_experiment
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        # Call enable_databricks_trace_archival multiple times
        result1 = enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")
        result2 = enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")
        result3 = enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # All calls should return the same archival location
    assert result1 == "catalog.schema.prefix_experiment_123_12345_genai_view"
    assert result2 == "catalog.schema.prefix_experiment_123_12345_genai_view"
    assert result3 == "catalog.schema.prefix_experiment_123_12345_genai_view"

    # Verify trace client was called 3 times (once per call)
    assert mock_trace_client_instance.create_trace_destination.call_count == 3

    # Verify view was recreated 3 times (once per call)
    assert mock_create_view.call_count == 3

    # Verify experiment tag was set 6 times (2 tags per call, 3 calls total)
    assert mock_client_instance.set_experiment_tag.call_count == 6

    # Verify both tags were set in each call
    # Storage table tag should be set 3 times
    storage_tag_calls = [
        call
        for call in mock_client_instance.set_experiment_tag.call_args_list
        if call[0][1] == MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE
    ]
    assert len(storage_tag_calls) == 3
    for call in storage_tag_calls:
        assert call[0] == (
            "12345",
            MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,
            "catalog.schema.prefix_experiment_123_12345_genai_view",
        )

    # Rolling deletion tag should be set 3 times
    rolling_deletion_calls = [
        call
        for call in mock_client_instance.set_experiment_tag.call_args_list
        if call[0][1] == MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED
    ]
    assert len(rolling_deletion_calls) == 3
    for call in rolling_deletion_calls:
        assert call[0] == ("12345", MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED, "true")


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_enablement_failure_due_to_storage_config_conflict(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test that ALREADY_EXISTS error is propagated when table schema versions change."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Mock trace client to raise ALREADY_EXISTS exception
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.side_effect = MlflowException(
        "ALREADY_EXISTS: Table schema version mismatch"
    )
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful MLflow client operations
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(MlflowException, match="ALREADY_EXISTS: Table schema version mismatch"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify view creation and tag setting were not called
    mock_create_view.assert_not_called()
    mock_client_instance.set_experiment_tag.assert_not_called()


# Rolling deletion failure tests


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_rolling_deletion_tag_failure(
    mock_mlflow_client,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test handling when setting the rolling deletion tag fails."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client
    # Create a valid config
    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        events_table_name="catalog.schema.events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client to return valid config
    mock_trace_client_instance = Mock()
    mock_trace_client_instance.create_trace_destination.return_value = mock_config
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful view creation
    mock_create_view.return_value = None

    # Mock client operations - storage tag succeeds, rolling deletion tag fails
    mock_client_instance = Mock()
    mock_experiment = Mock()
    mock_experiment.tags = {}
    mock_client_instance.get_experiment.return_value = mock_experiment

    # Configure set_experiment_tag to succeed for storage tag but fail for rolling deletion
    def side_effect(experiment_id, tag_key, tag_value):
        if tag_key == MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED:
            raise Exception("Failed to set rolling deletion tag")
        return None

    mock_client_instance.set_experiment_tag.side_effect = side_effect
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(MlflowException, match="Failed to enable trace rolling deletion"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify that storage tag was attempted before failure
    assert mock_client_instance.set_experiment_tag.call_count == 2
    mock_client_instance.set_experiment_tag.assert_any_call(
        "12345",
        MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,
        "catalog.schema.prefix_experiment_123_12345_genai_view",
    )


# Tests for set_experiment_storage_location function


@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._get_experiment_id")
def test_set_experiment_storage_location_unset_with_default_experiment(
    mock_get_experiment_id, mock_trace_client, mock_delta_archival_mixin
):
    """Test unsetting storage location with default experiment ID."""
    # Mock default experiment ID
    mock_get_experiment_id.return_value = "12345"

    # Mock trace client
    mock_trace_client_instance = Mock()
    mock_trace_client.return_value = mock_trace_client_instance

    # Setup initial cache state
    mock_delta_archival_mixin._config_cache["12345"] = "some_config"

    # Call with None location (unset)
    set_experiment_storage_location(None)

    # Verify delete was called with correct experiment ID
    mock_trace_client_instance.delete_trace_destination.assert_called_once_with("12345")

    # Verify cache was cleared
    assert "12345" not in mock_delta_archival_mixin._config_cache


@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
def test_set_experiment_storage_location_unset_with_explicit_experiment(
    mock_trace_client, mock_delta_archival_mixin
):
    """Test unsetting storage location with explicit experiment ID."""
    # Mock trace client
    mock_trace_client_instance = Mock()
    mock_trace_client.return_value = mock_trace_client_instance

    # Setup initial cache state
    mock_delta_archival_mixin._config_cache["67890"] = "some_config"

    # Call with None location and explicit experiment ID
    set_experiment_storage_location(None, experiment_id="67890")

    # Verify delete was called with correct experiment ID
    mock_trace_client_instance.delete_trace_destination.assert_called_once_with("67890")

    # Verify cache was cleared
    assert "67890" not in mock_delta_archival_mixin._config_cache


@patch("mlflow.genai.experimental.databricks_trace_archival.enable_databricks_trace_archival")
@patch("mlflow.genai.experimental.databricks_trace_archival._get_experiment_id")
def test_set_experiment_storage_location_set_with_default_experiment(
    mock_get_experiment_id, mock_enable_archival, mock_delta_archival_mixin
):
    """Test setting storage location with DatabricksUnityCatalog using default experiment ID."""
    # Mock default experiment ID
    mock_get_experiment_id.return_value = "67890"

    # Create a mock DatabricksUnityCatalog location
    location = DatabricksUnityCatalog(
        catalog="test_catalog", schema="test_schema", table_prefix="test_prefix"
    )

    # Setup initial cache state
    mock_delta_archival_mixin._config_cache["67890"] = "some_config"

    # Call with Unity Catalog location (no explicit experiment_id)
    set_experiment_storage_location(location)

    # Verify enable_databricks_trace_archival was called with default experiment ID
    mock_enable_archival.assert_called_once_with(
        "67890", "test_catalog", "test_schema", "test_prefix"
    )

    # Verify cache was cleared
    assert "67890" not in mock_delta_archival_mixin._config_cache


@patch("mlflow.genai.experimental.databricks_trace_archival.enable_databricks_trace_archival")
def test_set_experiment_storage_location_set_with_explicit_experiment(
    mock_enable_archival, mock_delta_archival_mixin
):
    """Test setting storage location with DatabricksUnityCatalog using explicit experiment ID."""
    # Create a mock DatabricksUnityCatalog location
    location = DatabricksUnityCatalog(
        catalog="test_catalog", schema="test_schema", table_prefix="test_prefix"
    )

    # Setup initial cache state
    mock_delta_archival_mixin._config_cache["12345"] = "some_config"

    # Call with Unity Catalog location and explicit experiment ID
    set_experiment_storage_location(location, experiment_id="12345")

    # Verify enable_databricks_trace_archival was called with explicit experiment ID
    mock_enable_archival.assert_called_once_with(
        "12345", "test_catalog", "test_schema", "test_prefix"
    )

    # Verify cache was cleared
    assert "12345" not in mock_delta_archival_mixin._config_cache


# Tests for re-create logic in enable_databricks_trace_archival


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival._create_genai_trace_view")
@patch("mlflow.genai.experimental.databricks_trace_archival._enable_trace_rolling_deletion")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_enable_archival_delete_on_already_exists_error(
    mock_mlflow_client,
    mock_rolling_deletion,
    mock_create_view,
    mock_trace_client,
    mock_workspace_client_class,
    mock_workspace_client,
):
    """Test that ALREADY_EXISTS error triggers delete and retry logic."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client

    # Create a valid config for the retry
    from mlflow.genai.experimental.databricks_trace_storage_config import (
        DatabricksTraceDeltaStorageConfig,
    )

    mock_config = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.prefix_12345_spans",
        events_table_name="catalog.schema.prefix_12345_events",
        spans_schema_version=SUPPORTED_SCHEMA_VERSION,
        events_schema_version=SUPPORTED_SCHEMA_VERSION,
    )

    # Mock trace client instance
    mock_trace_client_instance = Mock()

    # First call raises ALREADY_EXISTS, second call succeeds
    mock_error = Exception("Already exists")
    mock_error.error_code = "ALREADY_EXISTS"
    mock_trace_client_instance.create_trace_destination.side_effect = [
        mock_error,  # First call fails
        mock_config,  # Second call succeeds
    ]
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock successful client operations
    mock_client_instance = Mock()
    mock_experiment = Mock()
    mock_experiment.tags = {}  # No existing archival tag
    mock_client_instance.get_experiment.return_value = mock_experiment
    mock_client_instance.set_experiment_tag = Mock()  # Explicitly mock this
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        result = enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify delete was called after first failure
    mock_trace_client_instance.delete_trace_destination.assert_called_once_with("12345")

    # Verify create was called twice (first failed, second succeeded)
    assert mock_trace_client_instance.create_trace_destination.call_count == 2

    # Verify both calls had the same parameters
    expected_call_args = {
        "experiment_id": "12345",
        "catalog": "catalog",
        "schema": "schema",
        "table_prefix": "prefix",
    }

    for call in mock_trace_client_instance.create_trace_destination.call_args_list:
        assert call.kwargs == expected_call_args

    # Verify the rest of the flow completed successfully
    mock_create_view.assert_called_once()
    mock_client_instance.set_experiment_tag.assert_called()

    # Verify return value
    assert result == "catalog.schema.prefix_experiment_123_12345_genai_view"


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_enable_archival_non_already_exists_error_propagates(
    mock_mlflow_client, mock_trace_client, mock_workspace_client_class, mock_workspace_client
):
    """Test that non-ALREADY_EXISTS errors are propagated without delete/retry."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client

    # Mock trace client instance
    mock_trace_client_instance = Mock()

    # Create a non-ALREADY_EXISTS error
    mock_error = Exception("Permission denied")
    mock_error.error_code = "PERMISSION_DENIED"
    mock_trace_client_instance.create_trace_destination.side_effect = mock_error
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock MLflow client
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(Exception, match="Permission denied"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify delete was NOT called
    mock_trace_client_instance.delete_trace_destination.assert_not_called()

    # Verify create was only called once
    assert mock_trace_client_instance.create_trace_destination.call_count == 1


@patch("databricks.sdk.WorkspaceClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.DatabricksTraceServerClient")
@patch("mlflow.genai.experimental.databricks_trace_archival.MlflowClient")
def test_enable_archival_delete_retry_also_fails(
    mock_mlflow_client, mock_trace_client, mock_workspace_client_class, mock_workspace_client
):
    """Test that if the retry after delete also fails, the new error is propagated."""
    # Use the fixture's mock workspace client
    mock_workspace_client_class.return_value = mock_workspace_client

    # Mock trace client instance
    mock_trace_client_instance = Mock()

    # First call raises ALREADY_EXISTS, second call raises different error
    mock_already_exists_error = Exception("Already exists")
    mock_already_exists_error.error_code = "ALREADY_EXISTS"

    mock_retry_error = Exception("Invalid schema")
    mock_retry_error.error_code = "INVALID_PARAMETER_VALUE"

    mock_trace_client_instance.create_trace_destination.side_effect = [
        mock_already_exists_error,  # First call fails with ALREADY_EXISTS
        mock_retry_error,  # Second call fails with different error
    ]
    mock_trace_client.return_value = mock_trace_client_instance

    # Mock MLflow client
    mock_client_instance = Mock()
    mock_mlflow_client.return_value = mock_client_instance

    with patch("importlib.util.find_spec", return_value=Mock()):
        with pytest.raises(Exception, match="Invalid schema"):
            enable_databricks_trace_archival("12345", "catalog", "schema", "prefix")

    # Verify delete was called after first failure
    mock_trace_client_instance.delete_trace_destination.assert_called_once_with("12345")

    # Verify create was called twice
    assert mock_trace_client_instance.create_trace_destination.call_count == 2
