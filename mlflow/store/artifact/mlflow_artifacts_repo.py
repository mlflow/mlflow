from urllib.parse import urlparse
import posixpath
from collections import namedtuple
import requests

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
            f"The mlflow-artifacts uri was supplied with a port number: "
            f"{uri_parse.host}, but no host was defined."
        )


class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(self, artifact_uri):

        self._session = requests.Session()
        super().__init__(self.resolve_uri(artifact_uri))

    @classmethod
    def resolve_uri(cls, artifact_uri):
        tracking_uri = get_tracking_uri()

        track_parse = _parse_artifact_uri(tracking_uri)

        uri_parse = _parse_artifact_uri(artifact_uri)

        # Check to ensure that a port is present with no hostname
        _validate_port_mapped_to_hostname(uri_parse)

        api_path = "/api/2.0/mlflow-artifacts/artifacts"

        # If root directory is specified (empty path), `urllib.parse.urlparse` will pull
        # the api path from the uri. This logic is to handle this.
        if uri_parse.path != api_path:
            request_path = posixpath.join(api_path, uri_parse.path.lstrip("/"))
        else:
            request_path = api_path

        if uri_parse.host and uri_parse.port:
            resolved_artifacts_uri = (
                f"{track_parse.scheme}://{uri_parse.host}:{uri_parse.port}{request_path}"
            )
        elif uri_parse.host and not uri_parse.port:
            resolved_artifacts_uri = f"{track_parse.scheme}://{uri_parse.host}{request_path}"
        elif not uri_parse.host and not uri_parse.port:
            resolved_artifacts_uri = f"{tracking_uri}{request_path}"
        else:
            raise MlflowException(
                f"The supplied artifact uri {artifact_uri} could not be resolved."
            )

        return resolved_artifacts_uri.replace("///", "/").rstrip("/")
