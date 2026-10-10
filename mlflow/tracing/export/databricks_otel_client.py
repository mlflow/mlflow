"""Client for exporting MLflow spans to the Databricks OTLP ingest endpoint.

The OTel exporter owns batching and delivery classification. This module
owns local credential and workspace discovery, endpoint resolution, OTLP
serialization, OAuth token management, and the authenticated HTTP request.
"""

import json
import logging
import os
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta
from types import MethodType
from typing import Sequence

import requests

from mlflow.entities.span import Span
from mlflow.environment_variables import (
    MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT,
    MLFLOW_ENABLE_DB_SDK,
)
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH, build_otlp_export_request
from mlflow.utils.databricks_utils import _get_databricks_creds_config
from mlflow.utils.uri import get_db_info_from_uri

_logger = logging.getLogger(__name__)

# OTLP token path from databricks-sdk oauth module.  Keep this value local so
# importing the exporter does not require a particular SDK module layout.
_OIDC_TOKEN_PATH = "/oidc/v1/token"

_HEADER_AUTHORIZATION = "Authorization"
_HEADER_CONTENT_TYPE = "Content-Type"
_CONTENT_TYPE_PROTOBUF = "application/x-protobuf"
_COLLECTOR_TABLE_NAME_HEADER = "x-databricks-zerobus-table-name"

# Timeout (seconds) for an OAuth token mint or refresh.  The SDK's
# ``ClientCredentials`` implementation has historically called requests
# without a timeout, so the collector path supplies its own bounded refresh.
_TOKEN_REQUEST_TIMEOUT_SECONDS = 30

# Timeout (seconds) for one collector OTLP POST.
_REQUEST_TIMEOUT_SECONDS = 30

# Timeout for each collector metadata request.
_SDK_HTTP_TIMEOUT_SECONDS = 10.0

# Segment of the ingest hostname between the workspace ID and the region.
_COLLECTOR_HOST_SEGMENT = ".zerobus."

# OAuth resource for the table-scoped token accepted by the ingest service.
_COLLECTOR_TOKEN_RESOURCE = "api://databricks/workspaces/{workspace_id}/zerobusDirectWriteApi"

_VALID_COLLECTOR_HOST_SUFFIXES = (
    ".cloud.databricks.com",
    ".azuredatabricks.net",
    ".gcp.databricks.com",
)

_CLOUD_DOMAIN = {
    "aws": "cloud.databricks.com",
    "azure": "azuredatabricks.net",
    "gcp": "gcp.databricks.com",
}

# Configuration failures are unusual for qualified users and should produce a
# single warning per process.
_collector_config_failure_warned = False
_collector_config_failure_lock = threading.Lock()


class DatabricksOtelTokenError(RuntimeError):
    """A token could not be minted before a Databricks OTel request was sent."""


class DatabricksOtelTokenRefreshError(RuntimeError):
    """A Databricks OTel token could not be refreshed after a 401 response."""


class DatabricksOtelUnavailableError(RuntimeError):
    """The Databricks OTel collector is not applicable or available."""


class DatabricksOtelConfigurationError(DatabricksOtelUnavailableError):
    """A qualified Databricks OTel collector configuration could not be used."""


class DatabricksOtelSerializationError(RuntimeError):
    """An OTLP span batch could not be serialized for collector delivery."""


class DatabricksOTelClient:
    """Lazy, authenticated HTTP client for Databricks OTLP ingest."""

    # Resolved endpoints are shared by exporter/client instances because tracer
    # providers may be recreated during a process lifetime.
    _resolved_endpoints: dict[tuple[str, str], str] = {}
    _resolved_endpoints_lock = threading.Lock()

    def __init__(
        self,
        tracking_uri: str | None,
        token_source,
        table_name: str,
        host: str | None = None,
        workspace_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        # This constructor intentionally performs no credential, workspace, or
        # endpoint network lookup.  The first collector batch calls export_spans.
        self._tracking_uri = tracking_uri
        self._token_source = token_source
        self._table_name = table_name
        self._host = host
        self._workspace_id = str(workspace_id or "")
        self._client_id = client_id
        self._client_secret = client_secret
        self._endpoint_override = _normalize_collector_endpoint(endpoint) if endpoint else None
        self._collector_endpoint: str | None = None
        self._collector_url: str | None = None
        self._initialized = False
        self._initialization_error: DatabricksOtelUnavailableError | None = None
        self._initialization_lock = threading.Lock()
        self._session = requests.Session()

    def export_spans(self, spans: Sequence[Span]) -> requests.Response:
        """Serialize and export MLflow spans to the Databricks OTel collector."""
        self._ensure_initialized()

        try:
            request = build_otlp_export_request(list(spans))
        except Exception as exc:
            raise DatabricksOtelSerializationError(
                "Failed to build the Databricks OTel collector span export request"
            ) from exc

        try:
            payload = request.SerializeToString()
        except Exception as exc:
            raise DatabricksOtelSerializationError(
                "Failed to serialize the Databricks OTel collector span export request"
            ) from exc

        return self._post(payload)

    def close(self) -> None:
        """Close the client's HTTP connection pool."""
        self._session.close()

    def _initialize(self) -> None:
        configured_override = self._endpoint_override or (
            MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get()
        )

        # Keep an explicitly supplied token source and workspace together;
        # ambient credentials may belong to a different workspace.
        needs_credentials = not (self._host and self._workspace_id) or (
            self._token_source is None and not (self._client_id and self._client_secret)
        )
        if needs_credentials:
            credentials = _resolve_collector_credentials(self._tracking_uri)
            if credentials is None:
                raise DatabricksOtelUnavailableError(
                    "Databricks OTel collector credentials are unavailable"
                )
            self._host, self._workspace_id, self._client_id, self._client_secret = credentials

        if not self._workspace_id:
            raise DatabricksOtelUnavailableError(
                "Databricks OTel collector workspace configuration is unavailable"
            )

        if self._token_source is None:
            if not (self._host and self._client_id and self._client_secret):
                raise DatabricksOtelUnavailableError(
                    "Databricks OTel collector service-principal credentials are unavailable"
                )
            try:
                self._token_source = build_databricks_otel_collector_token_source(
                    host=self._host,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    workspace_id=self._workspace_id,
                    tables=[self._table_name],
                )
            except Exception as exc:
                _warn_collector_config_failure(
                    "Failed to build the Databricks OTel collector token source: %s", exc
                )
                raise DatabricksOtelConfigurationError(
                    "Failed to build the Databricks OTel collector token source"
                ) from exc

        if not configured_override and not self._host:
            raise DatabricksOtelUnavailableError("Databricks OTel collector host is unavailable")

        self._collector_endpoint = self._resolve_endpoint(configured_override)
        if self._collector_endpoint is None:
            raise DatabricksOtelConfigurationError(
                "Failed to resolve the Databricks OTel collector endpoint"
            )

        self._collector_url = f"https://{self._collector_endpoint}{OTLP_TRACES_PATH}"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if self._initialization_error is not None:
            raise self._initialization_error

        with self._initialization_lock:
            if self._initialized:
                return
            if self._initialization_error is not None:
                raise self._initialization_error

            try:
                self._initialize()
            except DatabricksOtelUnavailableError as exc:
                self._initialization_error = exc
                raise
            except Exception as exc:
                _warn_collector_config_failure(
                    "Failed to initialize the Databricks OTel collector client: %s", exc
                )
                error = DatabricksOtelConfigurationError(
                    "Failed to initialize the Databricks OTel collector client"
                )
                self._initialization_error = error
                raise error from exc
            self._initialized = True

    def _get_metastore_summary(self) -> dict[str, object]:
        """Read the workspace metastore summary using bounded HTTP requests."""
        client_id = self._client_id
        client_secret = self._client_secret
        if not (client_id and client_secret):
            raise ValueError("Service-principal credentials are required for collector discovery")

        host = self._host or ""
        token = _retrieve_token_with_timeout(
            client_id,
            client_secret,
            f"{host.rstrip('/')}{_OIDC_TOKEN_PATH}",
            {"grant_type": "client_credentials", "scope": "all-apis"},
            use_header=True,
        )
        response = requests.get(
            f"{host.rstrip('/')}/api/2.1/unity-catalog/metastore_summary",
            headers={
                "Accept": "application/json",
                "Authorization": f"{token.token_type} {token.access_token}",
                "X-Databricks-Workspace-Id": self._workspace_id,
            },
            timeout=_SDK_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        summary = response.json()
        if not isinstance(summary, dict):
            raise ValueError("Metastore summary is not a JSON object")
        return summary

    def _resolve_collector_endpoint_from_metastore(self) -> str | None:
        """Resolve a collector host from workspace metastore metadata."""
        try:
            summary = self._get_metastore_summary()
        except Exception as exc:
            _warn_collector_config_failure(
                "Failed to fetch metastore summary for collector endpoint resolution: %s", exc
            )
            return None

        region = summary.get("region")
        cloud_raw = summary.get("cloud")
        if not region:
            _warn_collector_config_failure(
                "Metastore summary returned empty region; cannot resolve collector endpoint."
            )
            return None

        if not cloud_raw:
            gid = str(summary.get("global_metastore_id") or "")
            parts = gid.split(":")
            cloud_raw = parts[0] if parts else ""

        cloud_raw = str(cloud_raw or "").lower()
        domain = _CLOUD_DOMAIN.get(cloud_raw)
        if not domain:
            _warn_collector_config_failure(
                "Unrecognised cloud %r from metastore summary; cannot resolve collector endpoint.",
                cloud_raw,
            )
            return None

        env_segment = "staging." if ".staging." in (self._host or "") else ""
        endpoint = f"{self._workspace_id}{_COLLECTOR_HOST_SEGMENT}{region}.{env_segment}{domain}"
        if not is_databricks_otel_collector_host(endpoint, self._workspace_id):
            _warn_collector_config_failure(
                "Assembled collector endpoint %r failed host validation.", endpoint
            )
            return None
        return _normalize_collector_endpoint(endpoint)

    def _resolve_endpoint(self, endpoint_override: str | None = None) -> str | None:
        """Resolve and validate the collector endpoint for this workspace."""
        if override := endpoint_override or MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get():
            if is_databricks_otel_collector_host(override, self._workspace_id):
                return _normalize_collector_endpoint(override)
            _warn_collector_config_failure(
                "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT override %r failed host validation "
                "for workspace_id=%r; ignoring override.",
                override,
                self._workspace_id,
            )
            return None

        cache_key = (self._host or "", self._workspace_id)
        with self._resolved_endpoints_lock:
            cached = self._resolved_endpoints.get(cache_key)
        if cached is not None:
            return _normalize_collector_endpoint(cached)

        resolved = self._resolve_collector_endpoint_from_metastore()
        if resolved is None:
            return None

        normalized = _normalize_collector_endpoint(resolved)
        with self._resolved_endpoints_lock:
            cached = self._resolved_endpoints.setdefault(cache_key, normalized)
        return _normalize_collector_endpoint(cached)

    def _force_token_refresh(self):
        """Mint a fresh token and update SDK token caches when available."""
        new_token = self._token_source.refresh()
        try:
            if hasattr(self._token_source, "_update_token"):
                with getattr(self._token_source, "_lock", nullcontext()):
                    self._token_source._update_token(new_token)
            elif hasattr(self._token_source, "_token"):
                with getattr(self._token_source, "_lock", nullcontext()):
                    self._token_source._token = new_token
        except Exception:
            _logger.debug(
                "Failed to cache the refreshed collector token on the token source; "
                "later exports will mint a new one.",
                exc_info=True,
            )
        return new_token

    def _post_with_token(self, payload: bytes, token) -> requests.Response:
        """POST one payload using the supplied token."""
        return self._session.post(
            self._collector_url,
            data=payload,
            headers={
                _HEADER_AUTHORIZATION: f"Bearer {token.access_token}",
                _HEADER_CONTENT_TYPE: _CONTENT_TYPE_PROTOBUF,
                _COLLECTOR_TABLE_NAME_HEADER: self._table_name,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    def _post(self, payload: bytes) -> requests.Response:
        """POST one payload, retrying exactly once after a 401 token response."""
        try:
            token = self._token_source.token()
        except Exception as exc:
            raise DatabricksOtelTokenError(
                "Minting the Databricks OTel collector token failed"
            ) from exc

        response = self._post_with_token(payload, token)
        if response.status_code != 401:
            return response

        try:
            refreshed = self._force_token_refresh()
        except Exception as exc:
            raise DatabricksOtelTokenRefreshError(
                "Refreshing the Databricks OTel collector token after HTTP 401 failed"
            ) from exc

        return self._post_with_token(payload, refreshed)


def _normalize_collector_endpoint(endpoint: str) -> str:
    """Return a collector host in the form used by the URL builder."""
    return endpoint.removeprefix("https://").rstrip("/")


def _retrieve_token_with_timeout(
    client_id: str,
    client_secret: str,
    token_url: str,
    params: dict[str, str],
    *,
    use_params: bool = False,
    use_header: bool = False,
):
    """Retrieve an OAuth token with a bounded HTTP request timeout."""
    from databricks.sdk.oauth import IgnoreNetrcAuth, Token

    request_params = dict(params)
    if use_params:
        if client_id:
            request_params["client_id"] = client_id
        if client_secret:
            request_params["client_secret"] = client_secret

    auth = (
        requests.auth.HTTPBasicAuth(client_id, client_secret) if use_header else IgnoreNetrcAuth()
    )
    response = requests.post(
        token_url,
        data=request_params,
        auth=auth,
        timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            error = response.json()
            code = error.get("errorCode", error.get("error", "unknown"))
            summary = error.get("errorSummary", error.get("error_description", "unknown"))
            summary = summary.replace("\r\n", " ")
            raise ValueError(f"{code}: {summary}")
        raise ValueError(response.content)

    try:
        body = response.json()
        expires_in = int(body["expires_in"])
        expiry = datetime.now() + timedelta(seconds=expires_in)
        return Token(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token"),
            token_type=body["token_type"],
            expiry=expiry,
        )
    except Exception as exc:
        raise NotImplementedError(f"Not supported yet: {exc}") from exc


def _bounded_client_credentials_refresh(token_source):
    """Refresh a databricks-sdk ``ClientCredentials`` source with a timeout."""
    params = {
        "grant_type": "client_credentials",
        "scope": "all-apis",
        "authorization_details": token_source._mlflow_authorization_details,
    }
    if token_source.endpoint_params:
        params.update(token_source.endpoint_params)
    return _retrieve_token_with_timeout(
        token_source.client_id,
        token_source.client_secret,
        token_source.token_url,
        params,
        use_params=token_source.use_params,
        use_header=token_source.use_header,
    )


def is_databricks_otel_collector_host(endpoint: str, workspace_id: str) -> bool:
    """Return whether *endpoint* is a valid workspace-scoped collector host."""
    host = endpoint.removeprefix("https://")
    if any(c in host for c in "/?#@:"):
        return False

    expected_prefix = f"{workspace_id}{_COLLECTOR_HOST_SEGMENT}"
    if not host.startswith(expected_prefix):
        return False

    remainder = host[len(expected_prefix) :]
    matched_suffix = next(
        (suffix for suffix in _VALID_COLLECTOR_HOST_SUFFIXES if remainder.endswith(suffix)),
        None,
    )
    if matched_suffix is None:
        return False

    region_part = remainder[: -len(matched_suffix)]
    return bool(region_part) and not region_part.endswith(".")


def _warn_collector_config_failure(reason: str, *args) -> None:
    """Warn once per process about a collector config or resolution failure."""
    global _collector_config_failure_warned
    message = reason + " Falling back to the MLflow tracing server span export path."
    with _collector_config_failure_lock:
        already_warned = _collector_config_failure_warned
        _collector_config_failure_warned = True
    if already_warned:
        _logger.debug(message, *args)
    else:
        _logger.warning(message, *args)


def build_table_authorization_details(tables: list[str]) -> str:
    """Return RFC 9396 authorization details for fully-qualified UC tables."""
    entries: list[dict[str, object]] = []
    for table in tables:
        parts = table.split(".")
        if len(parts) != 3:
            _logger.debug("Skipping malformed table name %r (expected cat.sch.tbl).", table)
            continue
        catalog, schema, _ = parts
        entries.extend([
            {
                "type": "unity_catalog_privileges",
                "object_type": "CATALOG",
                "object_full_path": catalog,
                "privileges": ["USE CATALOG"],
            },
            {
                "type": "unity_catalog_privileges",
                "object_type": "SCHEMA",
                "object_full_path": f"{catalog}.{schema}",
                "privileges": ["USE SCHEMA"],
            },
            {
                "type": "unity_catalog_privileges",
                "object_type": "TABLE",
                "object_full_path": table,
                "privileges": ["SELECT", "MODIFY"],
            },
        ])
    return json.dumps(entries)


def build_databricks_otel_collector_token_source(
    host: str,
    client_id: str,
    client_secret: str,
    workspace_id: str,
    tables: list[str],
):
    """Build a bounded, table-scoped databricks-sdk OAuth token source."""
    from databricks.sdk.oauth import ClientCredentials

    token_source = ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        token_url=f"{host.rstrip('/')}{_OIDC_TOKEN_PATH}",
        endpoint_params={"resource": _COLLECTOR_TOKEN_RESOURCE.format(workspace_id=workspace_id)},
        use_header=True,
    )
    token_source._mlflow_authorization_details = build_table_authorization_details(tables)
    token_source.refresh = MethodType(_bounded_client_credentials_refresh, token_source)
    return token_source


def _resolve_collector_credentials(tracking_uri: str | None):
    """Resolve local service-principal settings and a workspace ID lazily."""
    if not MLFLOW_ENABLE_DB_SDK.get():
        _logger.debug(
            "MLFLOW_ENABLE_DB_SDK is false; skipping Databricks OTel collector credential "
            "discovery."
        )
        return None

    try:
        config = _get_databricks_creds_config(tracking_uri)
        profile, _ = get_db_info_from_uri(tracking_uri or "")
    except Exception as exc:
        _logger.debug(
            "Failed to resolve local Databricks credentials for collector export: %s", exc
        )
        return None

    host = getattr(config, "host", None)
    client_id = getattr(config, "client_id", None)
    client_secret = getattr(config, "client_secret", None)
    auth_type = getattr(config, "auth_type", None)
    if auth_type and auth_type != "oauth-m2m":
        _logger.debug("Databricks profile selects %s authentication; using REST export.", auth_type)
        return None
    if auth_type != "oauth-m2m" and (
        getattr(config, "token", None)
        or (getattr(config, "username", None) and getattr(config, "password", None))
    ):
        _logger.debug("Databricks credentials include another auth method; using REST export.")
        return None
    if not (host and client_id and client_secret):
        _logger.debug(
            "Local Databricks credentials do not contain a host and service-principal pair."
        )
        return None

    workspace_id = (
        os.environ.get("DATABRICKS_WORKSPACE_ID")
        if not (profile or os.environ.get("DATABRICKS_CONFIG_PROFILE"))
        and os.environ.get("DATABRICKS_HOST", "").rstrip("/") == host.rstrip("/")
        else None
    )
    if not workspace_id:
        try:
            response = requests.get(
                f"{host.rstrip('/')}/.well-known/databricks-config",
                timeout=_SDK_HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            workspace_id = response.json().get("workspace_id")
        except Exception as exc:
            _logger.debug("Failed to resolve workspace ID for collector export: %s", exc)
            return None

    if not workspace_id:
        _logger.debug("Workspace ID is unavailable for collector export.")
        return None
    return host, str(workspace_id), client_id, client_secret
