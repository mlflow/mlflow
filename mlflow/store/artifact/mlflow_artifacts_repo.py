from urllib.parse import urlparse
from collections import namedtuple
import re

from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.exceptions import MlflowException


def _parse_artifact_uri(artifact_uri):
    ParsedURI = namedtuple("ParsedURI", "scheme host port path")
    parsed_uri = urlparse(artifact_uri)
    return ParsedURI(parsed_uri.scheme, parsed_uri.hostname, parsed_uri.port, parsed_uri.path)


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
    if uri_parse.host and _check_if_host_is_numeric(uri_parse.host) and not uri_parse.port:
        raise MlflowException(
            "The mlflow-artifacts uri was supplied with a port number: "
            f"{uri_parse.host}, but no host was defined."
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

        track_parse = _parse_artifact_uri(tracking_uri)

        uri_parse = _parse_artifact_uri(artifact_uri)

        # Check to ensure that a port is present with no hostname
        _validate_port_mapped_to_hostname(uri_parse)

        # Check that tracking uri is http or https
        _validate_uri_scheme(track_parse.scheme)

        if uri_parse.path == "/":  # root directory; build simple path
            resolved = f"{base_url}{uri_parse.path}"
        elif uri_parse.path == base_url:  # for operations like list artifacts
            resolved = base_url
        else:
            resolved = f"{base_url}/{track_parse.path}{uri_parse.path}"
        resolved = re.sub("//+", "/", resolved)

        if uri_parse.host and uri_parse.port:
            resolved_artifacts_uri = (
                f"{track_parse.scheme}://{uri_parse.host}:{uri_parse.port}{resolved}"
            )
        elif uri_parse.host and not uri_parse.port:
            resolved_artifacts_uri = f"{track_parse.scheme}://{uri_parse.host}{resolved}"
        elif not uri_parse.host and not uri_parse.port and uri_parse.path == track_parse.path:
            resolved_artifacts_uri = (
                f"{track_parse.scheme}://{track_parse.host}:{track_parse.port}{resolved}"
            )
        elif not uri_parse.host and not uri_parse.port:
            resolved_artifacts_uri = (
                f"{track_parse.scheme}://{track_parse.host}:" f"{track_parse.port}{resolved}"
            )
        else:
            raise MlflowException(
                f"The supplied artifact uri {artifact_uri} could not be resolved."
            )

        return resolved_artifacts_uri.replace("///", "/").rstrip("/")
