from urllib.parse import urlparse, urlunparse
import re

from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.exceptions import MlflowException


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


def _validate_uri_scheme(scheme):
    allowable_schemes = {"http", "https"}
    if scheme not in allowable_schemes:
        raise MlflowException(
            f"The configured tracking uri scheme: '{scheme}' is invalid for use with the proxy "
            f"mlflow-artifact scheme. The allowed tracking schemes are: {allowable_schemes}"
        )


class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(self, artifact_uri):
        super().__init__(self.resolve_uri(artifact_uri, get_tracking_uri()))

    @classmethod
    def resolve_uri(cls, artifact_uri, tracking_uri):
        base_url = "/api/2.0/mlflow-artifacts/artifacts"

        track_parse = urlparse(tracking_uri)

        uri_parse = urlparse(artifact_uri)

        # Check to ensure that a port is present with no hostname
        _validate_port_mapped_to_hostname(uri_parse)

        # Check that tracking uri is http or https
        _validate_uri_scheme(track_parse.scheme)

        if uri_parse.path == "/":  # root directory; build simple path
            resolved = f"{base_url}{uri_parse.path}"
        elif uri_parse.path == base_url:  # for operations like list artifacts
            resolved = base_url
        else:
            resolved = f"{track_parse.path}/{base_url}/{uri_parse.path}"
        resolved = re.sub("//+", "/", resolved)

        resolved_artifacts_uri = urlunparse(
            (
                # scheme
                track_parse.scheme,
                # netloc
                uri_parse.netloc if uri_parse.netloc else track_parse.netloc,
                # path
                resolved,
                # params
                "",
                # query
                "",
                # fragment
                "",
            )
        )

        return resolved_artifacts_uri.replace("///", "/").rstrip("/")
