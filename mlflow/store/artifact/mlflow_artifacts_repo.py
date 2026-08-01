import logging
import os
import re
import threading
from http import HTTPStatus
from urllib.parse import urlparse, urlunparse

from requests import HTTPError

from mlflow.environment_variables import (
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD,
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import http_request

_logger = logging.getLogger(__name__)

_SERVER_INFO_ENDPOINT = "/api/3.0/mlflow/server-info"
SERVER_INFO_MULTIPART_UPLOADS_ENABLED = "multipart_uploads_enabled"
SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED = "multipart_downloads_enabled"
# resolve_uri always embeds this service root; strip it to recover the deployment base URL
# used for /server-info (which lives beside /api/2.0, not under it).
_ARTIFACTS_SERVICE_ROOT = "/api/2.0/mlflow-artifacts/artifacts"


def _check_if_host_is_numeric(hostname):
    if hostname:
        try:
            float(hostname)
            return True
        except ValueError:
            return False
    else:
        return False


def _validate_port_mapped_to_hostname(uri_parse):
    # This check is to catch an mlflow-artifacts uri that has a port designated but no
    # hostname specified. `urllib.parse.urlparse` will treat such a uri as a filesystem
    # definition, mapping the provided port as a hostname value if this condition is not
    # validated.
    if uri_parse.hostname and _check_if_host_is_numeric(uri_parse.hostname) and not uri_parse.port:
        raise MlflowException(
            "The mlflow-artifacts uri was supplied with a port number: "
            f"{uri_parse.hostname}, but no host was defined."
        )


def _validate_uri_scheme(parsed_uri):
    allowable_schemes = {"http", "https"}
    if parsed_uri.scheme not in allowable_schemes:
        raise MlflowException(
            "When an mlflow-artifacts URI was supplied, the tracking URI must be a valid "
            f"http or https URI, but it was currently set to {parsed_uri.geturl()}. "
            "Perhaps you forgot to set the tracking URI to the running MLflow server. "
            "To set the tracking URI, use either of the following methods:\n"
            "1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. "
            "`export MLFLOW_TRACKING_URI=http://localhost:5000`\n"
            "2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. "
            "`mlflow.set_tracking_uri('http://localhost:5000')`"
        )


class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        effective_tracking_uri = tracking_uri or get_tracking_uri()
        super().__init__(
            self.resolve_uri(artifact_uri, effective_tracking_uri),
            tracking_uri=effective_tracking_uri,
            registry_uri=registry_uri,
        )
        self._server_capabilities: dict[str, bool] | None = None
        self._server_capabilities_lock = threading.Lock()

    @classmethod
    def resolve_uri(cls, artifact_uri, tracking_uri):
        base_url = "/api/2.0/mlflow-artifacts/artifacts"

        track_parse = urlparse(tracking_uri)

        uri_parse = urlparse(artifact_uri)

        # Check to ensure that a port is present with no hostname
        _validate_port_mapped_to_hostname(uri_parse)

        # Check that tracking uri is http or https
        _validate_uri_scheme(track_parse)

        if uri_parse.path == "/":  # root directory; build simple path
            resolved = f"{base_url}{uri_parse.path}"
        elif uri_parse.path == base_url:  # for operations like list artifacts
            resolved = base_url
        else:
            resolved = f"{track_parse.path}/{base_url}/{uri_parse.path}"
        resolved = re.sub(r"//+", "/", resolved)

        resolved_artifacts_uri = urlunparse((
            # scheme
            track_parse.scheme,
            # netloc
            uri_parse.netloc or track_parse.netloc,
            # path
            resolved,
            # params
            "",
            # query
            "",
            # fragment
            "",
        ))

        return resolved_artifacts_uri.replace("///", "/").rstrip("/")

    @property
    def _artifact_server_host_creds(self):
        """Credentials for the artifact-serving server root (not the artifact sub-path).

        Uses the resolved artifact URI host so `mlflow-artifacts://other-host/...` probes
        that host. Strips `/api/2.0/mlflow-artifacts/artifacts...` while preserving any tracking
        URI path prefix (e.g. `/mlflow`) so `/server-info` is requested at the correct base.
        """
        uri, _, _ = self.artifact_uri.partition(_ARTIFACTS_SERVICE_ROOT)
        return get_default_host_creds(uri.rstrip("/"))

    def _fetch_server_capabilities(self):
        """Fetch and cache multipart artifact capabilities from /server-info."""
        if self._server_capabilities is not None:
            return self._server_capabilities

        with self._server_capabilities_lock:
            if self._server_capabilities is not None:
                return self._server_capabilities

            try:
                response = http_request(
                    host_creds=self._artifact_server_host_creds,
                    endpoint=_SERVER_INFO_ENDPOINT,
                    method="GET",
                    timeout=3,
                    max_retries=0,
                    raise_on_status=False,
                )
                if response.status_code == 200:
                    data = response.json()
                    self._server_capabilities = {
                        SERVER_INFO_MULTIPART_UPLOADS_ENABLED: data.get(
                            SERVER_INFO_MULTIPART_UPLOADS_ENABLED, False
                        ),
                        SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED: data.get(
                            SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED, False
                        ),
                    }
                else:
                    _logger.debug(
                        "Failed to fetch multipart capabilities from %s (status=%s); "
                        "defaulting to disabled.",
                        _SERVER_INFO_ENDPOINT,
                        response.status_code,
                    )
                    self._server_capabilities = {}
            except Exception:
                _logger.debug(
                    "Failed to fetch multipart capabilities from %s; defaulting to disabled.",
                    _SERVER_INFO_ENDPOINT,
                    exc_info=True,
                )
                self._server_capabilities = {}

            return self._server_capabilities

    def _should_multipart_upload(self, local_file):
        # Check size first here so small mlflow-artifacts uploads skip the /server-info probe.
        return (
            os.path.getsize(local_file) >= MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get()
            and self._is_multipart_upload_enabled()
        )

    def _is_multipart_upload_enabled(self):
        if MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD.is_set():
            return MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD.get()
        return self._fetch_server_capabilities().get(SERVER_INFO_MULTIPART_UPLOADS_ENABLED, False)

    def _is_multipart_download_enabled(self):
        if MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.is_set():
            return MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.get()
        return self._fetch_server_capabilities().get(SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED, False)

    def _download_file(self, remote_file_path, local_path):
        if self._is_multipart_download_enabled():
            try:
                presigned_response = self._get_presigned_download_url(remote_file_path)
                file_size = presigned_response.file_size
                if file_size is not None:
                    chunk_size = MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get()
                    self._multipart_download(
                        presigned_response=presigned_response,
                        remote_file_path=remote_file_path,
                        local_path=local_path,
                        file_size=file_size,
                        chunk_size=chunk_size,
                    )
                    return
            except HTTPError as e:
                # When auto-detected via server-info, presigned failures indicate a
                # server misconfiguration — raise immediately.
                # When user forced via env var, fall back gracefully for legacy compat.
                if MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.is_set():
                    if e.response is not None and e.response.status_code in (
                        HTTPStatus.NOT_IMPLEMENTED,
                        HTTPStatus.NOT_FOUND,
                    ):
                        _logger.warning(
                            "Multipart download was requested but the server does not support "
                            "presigned downloads (HTTP %s). Falling back to proxied download. "
                            "Consider setting MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false to "
                            "avoid this warning and the extra presigned request on each download.",
                            e.response.status_code,
                        )
                    else:
                        raise
                else:
                    raise

        super()._download_file(remote_file_path, local_path)
