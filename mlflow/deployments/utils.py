import urllib
from urllib.parse import urlparse

from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path


def parse_target_uri(target_uri):
    """Parse out the deployment target from the provided target uri"""
    parsed = urllib.parse.urlparse(target_uri)
    if not parsed.scheme:
        if parsed.path:
            # uri = 'target_name' (without :/<path>)
            return parsed.path
        raise MlflowException(
            f"Not a proper deployment URI: {target_uri}. "
            + "Deployment URIs must be of the form 'target' or 'target:/suffix'"
        )
    return parsed.scheme


def _is_valid_uri(uri: str) -> bool:
    """
    Evaluates the basic structure of a provided uri to determine if the scheme and
    netloc are provided
    """
    try:
        parsed = urlparse(uri)
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False


def resolve_endpoint_url(base_url: str, endpoint: str) -> str:
    """
    Performs a validation on whether the returned value is a fully qualified url
    or requires the assembly of a fully qualified url by appending `endpoint`.

    :param base_url: The base URL. Should include the scheme and domain, e.g.,
                     ``http://127.0.0.1:6000``.
    :param endpoint: The endpoint to be appended to the base URL, e.g., ``/api/2.0/endpoints/`` or,
                     in the case of Databricks, the fully qualified url.
    :return: The complete URL, either directly returned or formed and returned by joining the
             base URL and the endpoint path.
    """
    return endpoint if _is_valid_uri(endpoint) else append_to_uri_path(base_url, endpoint)
