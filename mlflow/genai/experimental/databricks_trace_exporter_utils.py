"""
Utilities for Databricks trace archival functionality.

This module provides shared utilities for resolving Databricks ingest URLs
and other common archival operations.
"""

import logging
import re
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    GetTraceDestinationRequest,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceLocation as ProtoTraceLocation,
)
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint

_logger = logging.getLogger(__name__)


def create_archival_ingest_sdk():
    """
    Create a configured IngestApiSdk instance for trace archival.

    This function handles all the configuration resolution (ingest URL, workspace URL,
    and authentication token) and returns a ready-to-use IngestApiSdk instance.
    Environment variables can be used to override any of the resolved values.

    Returns:
        IngestApiSdk: Configured SDK instance ready for trace archival operations

    Raises:
        ImportError: If the ingest_api_sdk package is not available
        MlflowException: If configuration or authentication resolution fails

    Example:
        >>> sdk = create_archival_ingest_sdk()
        >>> # SDK is ready for creating streams and ingesting data
    """
    try:
        from ingest_api_sdk import IngestApiSdk  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The `databricks_ingest` package is required for trace archival. "
            "Please enroll in the preview by contacting your Databricks representative."
        )

    # Resolve all configuration components
    ingest_url = _resolve_ingest_url()
    workspace_url = _resolve_archival_workspace_url()
    token = _resolve_archival_token()

    # Create and return configured SDK instance
    _logger.debug(
        f"Creating IngestApiSdk with ingest URL: {ingest_url}, workspace URL: {workspace_url} "
    )
    return IngestApiSdk(ingest_url, workspace_url, token)


def _resolve_ingest_url(workspace_id: Optional[str] = None) -> str:
    """
    Dynamically resolve Databricks ingest URL from workspace host pattern.

    This function automatically determines the appropriate ingest URL based on the
    workspace host URL pattern and environment (dev/staging/prod) for both AWS and Azure.

    If MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL environment variable is set,
    it will be used as an override and returned immediately.

    TODO: This resolution logic should be part of the ingest_api_sdk and not in the client code here

    AWS Patterns:
    - Dev: *.dev.databricks.com → <workspace_id>.ingest.dev.cloud.databricks.com
    - Staging: *.staging.cloud.databricks.com → <workspace_id>.ingest.staging.cloud.databricks.com
    - Prod: *.cloud.databricks.com → <workspace_id>.ingest.cloud.databricks.com

    Azure Patterns:
    - Staging: *.staging.azuredatabricks.net → <workspace_id>.ingest.staging.azuredatabricks.net
    - Prod: *.azuredatabricks.net → <workspace_id>.ingest.azuredatabricks.net

    Args:
        workspace_id: Optional workspace ID. If not provided, will be auto-detected
            from Databricks context

    Returns:
        str: The resolved ingest URL

    Raises:
        MlflowException: If workspace_id cannot be determined or host pattern is unsupported

    Example:
        >>> ingest_url = resolve_ingest_url()
        >>> print(ingest_url)
        https://12345.ingest.staging.cloud.databricks.com
    """
    from mlflow.environment_variables import MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL
    from mlflow.exceptions import MlflowException
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    # Check for environment variable override first
    override_url = MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL.get()
    if override_url:
        _logger.debug(f"Using ingest URL from environment variable: {override_url}")
        return override_url

    # Get host credentials from Databricks context
    host_creds = get_databricks_host_creds()

    try:
        # Get workspace ID if not provided
        if workspace_id is None:
            from mlflow.utils.databricks_utils import get_workspace_id

            workspace_id = get_workspace_id()

        if not workspace_id:
            raise MlflowException(
                "Failed to resolve Databricks ingest URL: No workspace ID available. "
                "Ensure you are running in a Databricks environment or provide workspace_id "
                "parameter."
            )

        host_url = host_creds.host
        _logger.debug(f"Resolving ingest URL from host: {host_url}")

        # Remove protocol if present
        if host_url.startswith(("http://", "https://")):
            host_url = host_url.split("://", 1)[1]

        # Remove trailing slash and query parameters
        host_url = host_url.split("/")[0].split("?")[0]

        # AWS patterns - check in order of specificity
        if re.search(r"\.dev\.databricks\.com$", host_url):
            # AWS Dev: *.dev.databricks.com → workspace_id.ingest.dev.cloud.databricks.com
            ingest_url = f"https://{workspace_id}.ingest.dev.cloud.databricks.com"
            _logger.debug(f"Resolved AWS dev ingest URL: {ingest_url}")
            return ingest_url

        elif re.search(r"\.staging\.cloud\.databricks\.com$", host_url):
            # AWS Staging: *.staging.cloud.databricks.com →
            # workspace_id.ingest.staging.cloud.databricks.com
            ingest_url = f"https://{workspace_id}.ingest.staging.cloud.databricks.com"
            _logger.debug(f"Resolved AWS staging ingest URL: {ingest_url}")
            return ingest_url

        elif re.search(r"\.cloud\.databricks\.com$", host_url):
            # AWS Prod: *.cloud.databricks.com → workspace_id.ingest.cloud.databricks.com
            ingest_url = f"https://{workspace_id}.ingest.cloud.databricks.com"
            _logger.debug(f"Resolved AWS prod ingest URL: {ingest_url}")
            return ingest_url

        # Azure patterns
        elif re.search(r"\.staging\.azuredatabricks\.net$", host_url):
            # Azure Staging: *.staging.azuredatabricks.net →
            # workspace_id.ingest.staging.azuredatabricks.net
            ingest_url = f"https://{workspace_id}.ingest.staging.azuredatabricks.net"
            _logger.debug(f"Resolved Azure staging ingest URL: {ingest_url}")
            return ingest_url

        elif re.search(r"\.azuredatabricks\.net$", host_url):
            # Azure Prod: *.azuredatabricks.net → workspace_id.ingest.azuredatabricks.net
            ingest_url = f"https://{workspace_id}.ingest.azuredatabricks.net"
            _logger.debug(f"Resolved Azure prod ingest URL: {ingest_url}")
            return ingest_url

        else:
            # Unrecognized pattern
            error_msg = (
                f"Failed to resolve Databricks ingest URL: Unrecognized host pattern '{host_url}'. "
                f"Supported patterns: *.dev.databricks.com, *.staging.cloud.databricks.com, "
                f"*.cloud.databricks.com, *.staging.azuredatabricks.net, *.azuredatabricks.net"
            )
            _logger.error(error_msg)
            raise MlflowException(error_msg)

    except MlflowException:
        # Re-raise MlflowExceptions as-is
        raise
    except Exception as e:
        error_msg = f"Failed to resolve Databricks ingest URL: {e}"
        _logger.error(error_msg)
        raise MlflowException(error_msg) from e


def _resolve_archival_workspace_url() -> str:
    """
    Resolve the workspace URL for Databricks trace archival.

    This function returns the appropriate workspace URL for trace archival operations.
    If MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL environment variable is set,
    it will be used as an override and returned immediately. Otherwise, it returns
    the workspace URL from the Databricks host credentials.

    Returns:
        str: The resolved workspace URL

    Example:
        >>> workspace_url = resolve_archival_workspace_url()
        >>> print(workspace_url)
        https://my-workspace.cloud.databricks.com
    """
    from mlflow.environment_variables import MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    # Check for environment variable override first
    override_url = MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL.get()
    if override_url:
        _logger.debug(f"Using workspace URL from environment variable: {override_url}")
        return override_url

    # Get workspace URL from Databricks host credentials
    host_creds = get_databricks_host_creds()
    _logger.debug(f"Using workspace URL from host credentials: {host_creds.host}")
    return host_creds.host


def _resolve_archival_token() -> str:
    """
    Resolve the authentication token for Databricks trace archival.

    This function returns the appropriate authentication token for trace archival operations.
    If MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN environment variable is set,
    it will be used as an override and returned immediately. Otherwise, it returns
    the token from the Databricks host credentials.

    Returns:
        str: The resolved authentication token

    Raises:
        MlflowException: If no token is available from either source

    Example:
        >>> token = resolve_archival_token()
        >>> print(token[:10] + "...")  # Don't log full token
        dapi1234567...

    TODO: The token override is a stop gap until proper auth to ingestion is implemented.
    """
    from mlflow.environment_variables import MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN
    from mlflow.exceptions import MlflowException
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    # Check for environment variable override first
    override_token = MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN.get()
    if override_token:
        _logger.debug("Using authentication token from environment variable")
        return override_token

    # Get token from Databricks host credentials
    try:
        host_creds = get_databricks_host_creds()
        if not host_creds or not host_creds.token:
            raise MlflowException(
                "No Databricks authentication available for delta archival. "
                f"Either run in Databricks environment or set "
                f"{MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN.name}."
            )

        _logger.debug("Using authentication token from Databricks host credentials")
        return host_creds.token

    except Exception as e:
        if isinstance(e, MlflowException):
            raise
        else:
            raise MlflowException(
                f"Failed to resolve authentication token for delta archival: {e}"
            ) from e


class DatabricksTraceServerClient:
    """
    Client for interacting with Databricks Trace Server APIs.

    This client provides methods to create and retrieve trace destinations
    for archiving MLflow traces to Databricks Delta tables.
    """

    def __init__(self, host_creds=None):
        """Initialize the client with optional host credentials."""
        self._host_creds = host_creds or get_databricks_host_creds()

    def create_trace_destination(
        self, experiment_id: str, catalog: str, schema: str, table_prefix: Optional[str] = None
    ) -> DatabricksTraceDeltaStorageConfig:
        """
        Create a trace destination for archiving traces from an MLflow experiment.

        Args:
            experiment_id: The MLflow experiment ID
            catalog: The Unity Catalog catalog name
            schema: The Unity Catalog schema name
            table_prefix: Optional table prefix (defaults to server-generated)

        Returns:
            DatabricksTraceDeltaStorageConfig with the created destination info

        Raises:
            MlflowException: If creation fails (including ALREADY_EXISTS)
        """
        # Create proto request
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id

        proto_request = CreateTraceDestinationRequest(
            trace_location=proto_trace_location,
            uc_catalog=catalog,
            uc_schema=schema,
        )
        if table_prefix:
            proto_request.uc_table_prefix = table_prefix

        # Call the trace server API
        request_body = message_to_json(proto_request)

        response_proto = call_endpoint(
            host_creds=self._host_creds,
            endpoint="/api/2.0/tracing/trace-destinations",
            method="POST",
            json_body=request_body,
            response_proto=ProtoTraceDestination(),
        )

        # Convert response to config
        return self._proto_to_config(response_proto)

    def get_trace_destination(
        self, experiment_id: str
    ) -> Optional[DatabricksTraceDeltaStorageConfig]:
        """
        Get the trace destination configuration for an experiment.

        Args:
            experiment_id: The MLflow experiment ID

        Returns:
            DatabricksTraceDeltaStorageConfig if destination exists, None otherwise

        Raises:
            MlflowException: If there's an error (other than 404)
        """
        try:
            # Create proto request
            proto_trace_location = ProtoTraceLocation()
            proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
            proto_trace_location.mlflow_experiment.experiment_id = experiment_id

            proto_request = GetTraceDestinationRequest(
                trace_location=proto_trace_location,
            )

            # Call the trace server API
            request_body = message_to_json(proto_request)

            response_proto = call_endpoint(
                host_creds=self._host_creds,
                endpoint=f"/api/2.0/tracing/trace-destinations/mlflow-experiments/{experiment_id}",
                method="GET",
                json_body=request_body,
                response_proto=ProtoTraceDestination(),
            )

            # Convert response to config
            return self._proto_to_config(response_proto)

        except MlflowException as e:
            # Check if this is a 404 (not configured) vs other error
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            else:
                raise

    def _proto_to_config(self, proto: ProtoTraceDestination) -> DatabricksTraceDeltaStorageConfig:
        """Convert a TraceDestination proto to DatabricksTraceDeltaStorageConfig."""
        # Validate that this is an experiment location
        if proto.trace_location.type != ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT:
            raise MlflowException(
                f"TraceDestination only supports MLflow experiments, "
                f"but got location type: {proto.trace_location.type}"
            )

        if not proto.trace_location.mlflow_experiment:
            raise MlflowException(
                "TraceDestination requires an MLflow experiment location, "
                "but mlflow_experiment is None"
            )

        return DatabricksTraceDeltaStorageConfig(
            experiment_id=proto.trace_location.mlflow_experiment.experiment_id,
            spans_table_name=proto.spans_table_name,
            events_table_name=proto.events_table_name,
            spans_schema_version=proto.spans_schema_version,
            events_schema_version=proto.events_schema_version,
        )
