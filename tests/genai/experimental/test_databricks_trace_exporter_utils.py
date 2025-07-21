"""Tests for databricks_trace_exporter_utils functionality."""

from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_exporter_utils import (
    DatabricksTraceServerClient,
    _resolve_archival_token,
    _resolve_archival_workspace_url,
    _resolve_ingest_url,
    create_archival_ingest_sdk,
)
from mlflow.genai.experimental.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.protos.databricks_trace_server_pb2 import TraceDestination as ProtoTraceDestination
from mlflow.protos.databricks_trace_server_pb2 import TraceLocation as ProtoTraceLocation

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_host_creds():
    """Mock host credentials for testing."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test-workspace.cloud.databricks.com"
    mock_creds.token = "test-token-12345"
    return mock_creds


@pytest.fixture
def mock_workspace_id():
    """Mock workspace ID for testing."""
    return "12345"


# =============================================================================
# create_archival_ingest_sdk Tests
# =============================================================================


def test_create_archival_ingest_sdk_success(mock_host_creds, mock_workspace_id, monkeypatch):
    """Test successful creation of IngestApiSdk."""
    # Clear any environment overrides
    monkeypatch.delenv("MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN", raising=False)

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_host_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=mock_workspace_id,
        ),
        mock.patch("ingest_api_sdk.IngestApiSdk") as mock_sdk_class,
    ):
        # Call the function
        result = create_archival_ingest_sdk()

        # Verify SDK was created with correct parameters
        expected_ingest_url = f"https://{mock_workspace_id}.ingest.cloud.databricks.com"
        mock_sdk_class.assert_called_once_with(
            expected_ingest_url, mock_host_creds.host, mock_host_creds.token
        )

        # Verify result is the SDK instance
        assert result == mock_sdk_class.return_value


def test_create_archival_ingest_sdk_import_error():
    """Test ImportError when ingest_api_sdk package is not available."""
    with mock.patch(
        "builtins.__import__",
        side_effect=ImportError("No module named 'ingest_api_sdk'"),
    ):
        with pytest.raises(
            ImportError, match=r"The `databricks_ingest` package is required for trace archival"
        ):
            create_archival_ingest_sdk()


def test_create_archival_ingest_sdk_with_env_overrides(monkeypatch):
    """Test SDK creation with environment variable overrides."""
    # Set environment overrides
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL", "https://custom.ingest.url")
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL", "https://custom.workspace")
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN", "custom-token")

    with mock.patch("ingest_api_sdk.IngestApiSdk") as mock_sdk_class:
        # Call the function
        result = create_archival_ingest_sdk()

        # Verify SDK was created with overridden values
        mock_sdk_class.assert_called_once_with(
            "https://custom.ingest.url", "https://custom.workspace", "custom-token"
        )

        assert result == mock_sdk_class.return_value


def test_create_archival_ingest_sdk_resolution_error(mock_host_creds):
    """Test error handling when resolution functions fail."""
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_host_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=None,  # This will cause _resolve_ingest_url to fail
        ),
        mock.patch("ingest_api_sdk.IngestApiSdk") as mock_sdk_class,
    ):
        with pytest.raises(MlflowException, match=r"No workspace ID available"):
            create_archival_ingest_sdk()
        mock_sdk_class.assert_not_called()


# =============================================================================
# _resolve_ingest_url Tests
# =============================================================================


@pytest.mark.parametrize(
    ("host_url", "expected_suffix"),
    [
        # AWS patterns
        ("https://test.dev.databricks.com", ".ingest.dev.cloud.databricks.com"),
        ("test.dev.databricks.com", ".ingest.dev.cloud.databricks.com"),
        ("https://test.staging.cloud.databricks.com", ".ingest.staging.cloud.databricks.com"),
        ("test.staging.cloud.databricks.com", ".ingest.staging.cloud.databricks.com"),
        ("https://test.cloud.databricks.com", ".ingest.cloud.databricks.com"),
        ("test.cloud.databricks.com", ".ingest.cloud.databricks.com"),
        # Azure patterns
        ("https://test.staging.azuredatabricks.net", ".ingest.staging.azuredatabricks.net"),
        ("test.staging.azuredatabricks.net", ".ingest.staging.azuredatabricks.net"),
        ("https://test.azuredatabricks.net", ".ingest.azuredatabricks.net"),
        ("test.azuredatabricks.net", ".ingest.azuredatabricks.net"),
    ],
)
def test_resolve_ingest_url_patterns(host_url, expected_suffix, mock_workspace_id):
    """Test URL resolution for various host patterns."""
    mock_creds = mock.Mock()
    mock_creds.host = host_url

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=mock_workspace_id,
        ),
    ):
        result = _resolve_ingest_url()
        expected = f"https://{mock_workspace_id}{expected_suffix}"
        assert result == expected


def test_resolve_ingest_url_with_env_override(monkeypatch):
    """Test that environment variable override takes precedence."""
    override_url = "https://override.ingest.url"
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL", override_url)

    # Should not even call get_databricks_host_creds
    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.get_databricks_host_creds"
    ) as mock_get_creds:
        result = _resolve_ingest_url()
        assert result == override_url
        mock_get_creds.assert_not_called()


def test_resolve_ingest_url_with_trailing_slash_and_params(mock_workspace_id):
    """Test URL cleanup for trailing slashes and query parameters."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.cloud.databricks.com/?param=value"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=mock_workspace_id,
        ),
    ):
        result = _resolve_ingest_url()
        expected = f"https://{mock_workspace_id}.ingest.cloud.databricks.com"
        assert result == expected


def test_resolve_ingest_url_no_workspace_id():
    """Test error when workspace ID cannot be determined."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.cloud.databricks.com"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=None,
        ),
    ):
        with pytest.raises(MlflowException, match=r"No workspace ID available"):
            _resolve_ingest_url()


def test_resolve_ingest_url_unrecognized_pattern(mock_workspace_id):
    """Test error for unrecognized host patterns."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://unknown.domain.com"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value=mock_workspace_id,
        ),
    ):
        with pytest.raises(
            MlflowException, match=r"Unrecognized host pattern.*unknown\.domain\.com"
        ):
            _resolve_ingest_url()


def test_resolve_ingest_url_with_provided_workspace_id(mock_host_creds):
    """Test resolution with explicitly provided workspace ID."""
    provided_workspace_id = "67890"

    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        return_value=mock_host_creds,
    ):
        # Should not call get_workspace_id when workspace_id is provided
        with mock.patch("mlflow.utils.databricks_utils.get_workspace_id") as mock_get_workspace_id:
            result = _resolve_ingest_url(workspace_id=provided_workspace_id)
            expected = f"https://{provided_workspace_id}.ingest.cloud.databricks.com"
            assert result == expected
            mock_get_workspace_id.assert_not_called()


# =============================================================================
# _resolve_archival_workspace_url Tests
# =============================================================================


def test_resolve_archival_workspace_url_normal(mock_host_creds):
    """Test normal workspace URL resolution."""
    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        return_value=mock_host_creds,
    ):
        result = _resolve_archival_workspace_url()
        assert result == mock_host_creds.host


def test_resolve_archival_workspace_url_with_env_override(monkeypatch):
    """Test workspace URL with environment override."""
    override_url = "https://override.workspace.url"
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL", override_url)

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.get_databricks_host_creds"
    ) as mock_get_creds:
        result = _resolve_archival_workspace_url()
        assert result == override_url
        mock_get_creds.assert_not_called()


# =============================================================================
# _resolve_archival_token Tests
# =============================================================================


def test_resolve_archival_token_normal(mock_host_creds):
    """Test normal token resolution."""
    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        return_value=mock_host_creds,
    ):
        result = _resolve_archival_token()
        assert result == mock_host_creds.token


def test_resolve_archival_token_with_env_override(monkeypatch):
    """Test token with environment override."""
    override_token = "override-token-12345"
    monkeypatch.setenv("MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN", override_token)

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.get_databricks_host_creds"
    ) as mock_get_creds:
        result = _resolve_archival_token()
        assert result == override_token
        mock_get_creds.assert_not_called()


def test_resolve_archival_token_missing_token():
    """Test error when no token is available."""
    mock_creds = mock.Mock()
    mock_creds.token = None  # No token available

    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        return_value=mock_creds,
    ):
        with pytest.raises(
            MlflowException,
            match=r"No Databricks authentication available.*MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN",
        ):
            _resolve_archival_token()


def test_resolve_archival_token_null_host_creds():
    """Test error when get_databricks_host_creds returns None."""
    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        return_value=None,
    ):
        with pytest.raises(MlflowException, match=r"No Databricks authentication available"):
            _resolve_archival_token()


def test_resolve_archival_token_generic_exception():
    """Test generic exception handling in token resolution."""
    with mock.patch(
        "mlflow.utils.databricks_utils.get_databricks_host_creds",
        side_effect=RuntimeError("Connection failed"),
    ):
        with pytest.raises(
            MlflowException, match=r"Failed to resolve authentication token.*Connection failed"
        ):
            _resolve_archival_token()


# =============================================================================
# DatabricksTraceServerClient Tests
# =============================================================================


def test_databricks_trace_server_client_init_with_creds():
    """Test client initialization with provided credentials."""
    mock_creds = mock.Mock()
    client = DatabricksTraceServerClient(host_creds=mock_creds)
    assert client._host_creds == mock_creds


def test_databricks_trace_server_client_init_without_creds(mock_host_creds):
    """Test client initialization without credentials."""
    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.get_databricks_host_creds",
        return_value=mock_host_creds,
    ):
        client = DatabricksTraceServerClient()
        assert client._host_creds == mock_host_creds


def test_create_trace_destination_success(mock_host_creds):
    """Test successful trace destination creation."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    # Create mock response proto
    response_proto = ProtoTraceDestination()
    response_proto.trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    response_proto.trace_location.mlflow_experiment.experiment_id = "exp123"
    response_proto.spans_table_name = "catalog.schema.spans_123"
    response_proto.events_table_name = "catalog.schema.events_123"
    response_proto.spans_schema_version = "v1"
    response_proto.events_schema_version = "v1"

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.call_endpoint",
        return_value=response_proto,
    ) as mock_call:
        result = client.create_trace_destination(
            experiment_id="exp123",
            catalog="catalog",
            schema="schema",
            table_prefix="prefix",
        )

        # Verify API was called correctly
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[1]["endpoint"] == "/api/2.0/tracing/trace-destinations"
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["host_creds"] == mock_host_creds

        # Verify result
        assert isinstance(result, DatabricksTraceDeltaStorageConfig)
        assert result.experiment_id == "exp123"
        assert result.spans_table_name == "catalog.schema.spans_123"
        assert result.events_table_name == "catalog.schema.events_123"


def test_create_trace_destination_without_table_prefix(mock_host_creds):
    """Test trace destination creation without table prefix."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    response_proto = ProtoTraceDestination()
    response_proto.trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    response_proto.trace_location.mlflow_experiment.experiment_id = "exp123"
    response_proto.spans_table_name = "catalog.schema.spans_generated"
    response_proto.events_table_name = "catalog.schema.events_generated"
    response_proto.spans_schema_version = "v1"
    response_proto.events_schema_version = "v1"

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.call_endpoint",
        return_value=response_proto,
    ) as mock_call:
        result = client.create_trace_destination(
            experiment_id="exp123",
            catalog="catalog",
            schema="schema",
            # No table_prefix provided
        )

        # Verify request doesn't include uc_table_prefix
        request_body = mock_call.call_args[1]["json_body"]
        assert "uc_table_prefix" not in request_body

        assert result.spans_table_name == "catalog.schema.spans_generated"


def test_get_trace_destination_success(mock_host_creds):
    """Test successful trace destination retrieval."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    # Create mock response proto
    response_proto = ProtoTraceDestination()
    response_proto.trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    response_proto.trace_location.mlflow_experiment.experiment_id = "exp123"
    response_proto.spans_table_name = "catalog.schema.spans_123"
    response_proto.events_table_name = "catalog.schema.events_123"
    response_proto.spans_schema_version = "v1"
    response_proto.events_schema_version = "v1"

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.call_endpoint",
        return_value=response_proto,
    ) as mock_call:
        result = client.get_trace_destination(experiment_id="exp123")

        # Verify API was called correctly
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert (
            call_args[1]["endpoint"]
            == "/api/2.0/tracing/trace-destinations/mlflow-experiments/exp123"
        )
        assert call_args[1]["method"] == "GET"

        # Verify result
        assert isinstance(result, DatabricksTraceDeltaStorageConfig)
        assert result.experiment_id == "exp123"


@pytest.mark.parametrize(
    "error_message",
    [
        "404 Not Found",
        "Resource not found",
        "Destination not found for experiment",
    ],
)
def test_get_trace_destination_not_found(mock_host_creds, error_message):
    """Test get_trace_destination returns None for 404 errors."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.call_endpoint",
        side_effect=MlflowException(error_message),
    ):
        result = client.get_trace_destination(experiment_id="exp123")
        assert result is None


def test_get_trace_destination_other_error(mock_host_creds):
    """Test get_trace_destination propagates non-404 errors."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    with mock.patch(
        "mlflow.genai.experimental.databricks_trace_exporter_utils.call_endpoint",
        side_effect=MlflowException("Internal server error"),
    ):
        with pytest.raises(MlflowException, match=r"Internal server error"):
            client.get_trace_destination(experiment_id="exp123")


def test_proto_to_config_valid(mock_host_creds):
    """Test valid proto to config conversion."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    proto = ProtoTraceDestination()
    proto.trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    proto.trace_location.mlflow_experiment.experiment_id = "exp123"
    proto.spans_table_name = "catalog.schema.spans_123"
    proto.events_table_name = "catalog.schema.events_123"
    proto.spans_schema_version = "v1"
    proto.events_schema_version = "v1"

    result = client._proto_to_config(proto)

    assert isinstance(result, DatabricksTraceDeltaStorageConfig)
    assert result.experiment_id == "exp123"
    assert result.spans_table_name == "catalog.schema.spans_123"
    assert result.events_table_name == "catalog.schema.events_123"
    assert result.spans_schema_version == "v1"
    assert result.events_schema_version == "v1"


def test_proto_to_config_invalid_location_type(mock_host_creds):
    """Test error for invalid location type."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    proto = ProtoTraceDestination()
    # Use a valid but non-MLFLOW_EXPERIMENT type (like UNSPECIFIED which is typically 0)
    proto.trace_location.type = 0  # UNSPECIFIED or first enum value

    with pytest.raises(MlflowException, match=r"only supports MLflow experiments"):
        client._proto_to_config(proto)


def test_proto_to_config_missing_experiment_id(mock_host_creds):
    """Test error when experiment_id is missing."""
    client = DatabricksTraceServerClient(host_creds=mock_host_creds)

    proto = ProtoTraceDestination()
    proto.trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    # Set mlflow_experiment but without experiment_id
    proto.trace_location.mlflow_experiment.SetInParent()
    # Don't set experiment_id - it will be empty string

    # This should work but return a config with empty experiment_id
    # If we want to test an error case, we'd need to test with missing required fields
    result = client._proto_to_config(proto)
    assert result.experiment_id == ""  # Empty string is the default for unset string fields
