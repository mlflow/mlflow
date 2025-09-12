"""
Utilities for Databricks trace archival functionality.

This module provides shared utilities for resolving Databricks ingest URLs
and other common archival operations.
"""

import logging
import re
import socket
import subprocess

from google.protobuf.empty_pb2 import Empty

from mlflow.entities.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    DeleteTraceDestinationRequest,
    GetTraceDestinationRequest,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceLocation as ProtoTraceLocation,
)
from mlflow.utils.databricks_utils import get_databricks_host_creds, get_workspace_url
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint

_logger = logging.getLogger(__name__)


def _get_workspace_id():
    """Get workspace ID from Databricks SDK."""
    try:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient().get_workspace_id()
    except ImportError as e:
        raise ImportError("databricks-sdk is required for trace archival functionality") from e


def _get_host():
    """Get host from Databricks context."""
    host = get_workspace_url()
    if not host:
        host_creds = get_databricks_host_creds()
        host = host_creds.host
    return host


def _extract_region_from_host(host_url: str) -> str:
    """
    Extract AWS region from workspace hostname via DNS lookup.

    Performs DNS resolution on the workspace hostname to get the load balancer
    hostname, then extracts the region from patterns like:
    public-ingress-*.elb.us-east-1.amazonaws.com -> us-east-1

    Args:
        host_url: The workspace URL (may include protocol and paths)

    Returns:
        str: The extracted region (e.g., 'us-east-1')

    Raises:
        MlflowException: If DNS lookup fails or region cannot be extracted
    """
    from mlflow.exceptions import MlflowException

    # Clean the hostname - remove protocol and paths
    hostname = host_url
    if hostname.startswith(("http://", "https://")):
        hostname = hostname.split("://", 1)[1]
    hostname = hostname.split("/")[0].split("?")[0]

    _logger.debug(f"Performing DNS lookup for hostname: {hostname}")

    # Try nslookup first as the default approach
    try:
        # Use nslookup command as primary approach
        result = subprocess.run(["nslookup", hostname], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Parse nslookup output for CNAME or canonical name
            for line in result.stdout.split("\n"):
                if "canonical name" in line.lower() or "cname" in line.lower():
                    # Extract the canonical name
                    parts = line.split()
                    if len(parts) >= 4:
                        canonical_name = parts[-1].rstrip(".")
                        region_match = re.search(r"\.elb\.([^.]+)\.amazonaws\.com$", canonical_name)
                        if region_match:
                            region = region_match.group(1)
                            _logger.debug(
                                f"Extracted region from nslookup canonical name: {region}"
                            )
                            return region

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        # nslookup command failed or not available, fall back to socket-based approach
        _logger.debug(f"nslookup failed ({e}), falling back to socket-based DNS lookup")

    # Fallback to socket-based DNS resolution
    try:
        _, _, ip_addresses = socket.gethostbyname_ex(hostname)

        # For workspace hostnames, we need to resolve to get the load balancer info
        # Try to get canonical name through reverse DNS
        for ip in ip_addresses:
            try:
                reverse_hostname, _, _ = socket.gethostbyaddr(ip)
                _logger.debug(f"Reverse DNS for {ip}: {reverse_hostname}")

                # Look for AWS ELB pattern: *.elb.<region>.amazonaws.com
                region_match = re.search(r"\.elb\.([^.]+)\.amazonaws\.com$", reverse_hostname)
                if region_match:
                    region = region_match.group(1)
                    _logger.debug(f"Extracted region from load balancer hostname: {region}")
                    return region

            except socket.herror:
                # Reverse DNS failed for this IP, try next one
                continue

        # If we still haven't found a region, raise an error
        error_msg = (
            f"Failed to extract region from workspace hostname '{hostname}'. "
            f"Could not find AWS load balancer pattern in DNS resolution results."
        )
        _logger.error(error_msg)
        raise MlflowException(error_msg)

    except socket.gaierror as e:
        error_msg = f"DNS resolution failed for hostname '{hostname}': {e}"
        _logger.error(error_msg)
        raise MlflowException(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to extract region from hostname '{hostname}': {e}"
        _logger.error(error_msg)
        raise MlflowException(error_msg) from e


def _extract_domain_from_host(host_url: str) -> str:
    """
    Extract the domain from a workspace URL using specific pattern matching.

    Args:
        host_url: The workspace URL (may include protocol and paths)

    Returns:
        str: The extracted domain

    Raises:
        MlflowException: If the host pattern is not recognized
    """
    from mlflow.exceptions import MlflowException

    # Clean the hostname - remove protocol and paths
    hostname = host_url
    if hostname.startswith(("http://", "https://")):
        hostname = hostname.split("://", 1)[1]
    hostname = hostname.split("/")[0].split("?")[0]

    _logger.debug(f"Extracting domain from hostname: {hostname}")

    # AWS patterns - check in order of specificity
    if re.search(r"\.dev\.databricks\.com$", hostname):
        domain = "dev.databricks.com"
        _logger.debug(f"Matched AWS dev pattern, domain: {domain}")
        return domain

    elif re.search(r"\.staging\.cloud\.databricks\.com$", hostname):
        domain = "staging.cloud.databricks.com"
        _logger.debug(f"Matched AWS staging pattern, domain: {domain}")
        return domain

    elif re.search(r"\.cloud\.databricks\.com$", hostname):
        domain = "cloud.databricks.com"
        _logger.debug(f"Matched AWS prod pattern, domain: {domain}")
        return domain

    # Azure patterns
    elif re.search(r"\.staging\.azuredatabricks\.net$", hostname):
        domain = "staging.azuredatabricks.net"
        _logger.debug(f"Matched Azure staging pattern, domain: {domain}")
        return domain

    elif re.search(r"\.azuredatabricks\.net$", hostname):
        domain = "azuredatabricks.net"
        _logger.debug(f"Matched Azure prod pattern, domain: {domain}")
        return domain

    else:
        # Unrecognized pattern
        error_msg = (
            f"Failed to extract domain: Unrecognized host pattern '{hostname}'. "
            f"Supported patterns: *.dev.databricks.com, *.staging.cloud.databricks.com, "
            f"*.cloud.databricks.com, *.staging.azuredatabricks.net, *.azuredatabricks.net"
        )
        _logger.error(error_msg)
        raise MlflowException(error_msg)


def create_archival_zerobus_sdk():
    """
    Create a configured ZerobusSdk instance for trace archival.

    This function handles all the configuration resolution (ingest URL, workspace URL,
    and authentication token) and returns a ready-to-use ZerobusSdk instance.
    Environment variables can be used to override any of the resolved values.

    Returns:
        ZerobusSdk: Configured SDK instance ready for trace archival operations

    Raises:
        ImportError: If the zerobus_sdk package is not available
        MlflowException: If configuration or authentication resolution fails

    Example:
        >>> sdk = create_archival_zerobus_sdk()
        >>> # SDK is ready for creating streams and ingesting data
    """
    try:
        from zerobus_sdk import ZerobusSdk  # type: ignore[import-untyped]
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
        f"Creating ZerobusSdk with ingest URL: {ingest_url}, workspace URL: {workspace_url} "
    )
    return ZerobusSdk(ingest_url, workspace_url, token)


def _resolve_ingest_url() -> str:
    """
    Resolve Databricks ingest URL using DNS-based region extraction.

    This function generates ingest URLs in the format:
    <workspace_id>.zerobus.<region>.<domain>

    The region is extracted by performing DNS lookup on the workspace hostname
    to find the load balancer hostname pattern (e.g., *.elb.us-east-1.amazonaws.com).

    If MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL environment variable is set,
    it will be used as an override and returned immediately.

    Returns:
        str: The resolved ingest URL

    Raises:
        MlflowException: If workspace_id, region, or domain cannot be determined

    Example:
        >>> ingest_url = _resolve_ingest_url()
        >>> print(ingest_url)
        123.zerobus.us-west-2.cloud.databricks.com
    """
    from mlflow.environment_variables import MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL
    from mlflow.exceptions import MlflowException

    # Check for environment variable override first
    override_url = MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL.get()
    if override_url:
        _logger.debug(f"Using ingest URL from environment variable: {override_url}")
        return override_url

    try:
        # Get all required components
        workspace_id = _get_workspace_id()
        if not workspace_id:
            raise MlflowException(
                "Failed to resolve Databricks ingest URL: No workspace ID available. "
                "Ensure you are running in a Databricks environment."
            )

        host_url = _get_host()
        _logger.debug(f"Resolving ingest URL from host: {host_url}")

        # Extract region via DNS lookup
        region = _extract_region_from_host(host_url)

        # Extract domain from host URL
        domain = _extract_domain_from_host(host_url)

        # Build ingest URL in new format
        ingest_url = f"{workspace_id}.zerobus.{region}.{domain}"
        _logger.debug(f"Resolved ingest URL: {ingest_url}")
        return ingest_url

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
        my-workspace.cloud.databricks.com
    """
    from mlflow.environment_variables import MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL

    # Check for environment variable override first
    override_url = MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL.get()
    if override_url:
        _logger.debug(f"Using workspace URL from environment variable: {override_url}")
        return override_url

    # Get workspace URL from Databricks context
    workspace_url = _get_host()

    # Remove protocol if present to ensure consistency with ingest URL format
    if workspace_url.startswith(("http://", "https://")):
        workspace_url = workspace_url.split("://", 1)[1]

    # Remove trailing slash and query parameters
    workspace_url = workspace_url.split("/")[0].split("?")[0]

    _logger.debug(f"Using workspace URL from host credentials: {workspace_url}")
    return workspace_url


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
        self, experiment_id: str, catalog: str, schema: str, table_prefix: str | None = None
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

    def get_trace_destination(self, experiment_id: str) -> DatabricksTraceDeltaStorageConfig | None:
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

    def delete_trace_destination(self, experiment_id: str) -> None:
        """
        Delete the trace destination configuration for an experiment.

        Args:
            experiment_id: The MLflow experiment ID

        Returns:
            None

        Raises:
            MlflowException: If deletion fails
        """
        # Create proto request
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id

        proto_request = DeleteTraceDestinationRequest(
            trace_location=proto_trace_location,
        )

        # Call the trace server API
        request_body = message_to_json(proto_request)

        call_endpoint(
            host_creds=self._host_creds,
            endpoint=f"/api/2.0/tracing/trace-destinations/mlflow-experiments/{experiment_id}",
            method="DELETE",
            json_body=request_body,
            response_proto=Empty(),
        )

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
            logs_table_name=proto.logs_table_name,
            spans_schema_version=proto.spans_schema_version,
            logs_schema_version=proto.logs_schema_version,
        )
