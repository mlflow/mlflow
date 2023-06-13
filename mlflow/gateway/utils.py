from functools import wraps
import logging
import psutil
import re
from requests import HTTPError
from typing import Optional, List
from urllib.parse import urljoin, urlparse

from mlflow.environment_variables import (
    MLFLOW_GATEWAY_URI,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_HEALTH_ENDPOINT
from mlflow.utils import env
from mlflow.utils.request_utils import (
    augmented_raise_for_status,
    _get_http_response_with_retries,
    _TRANSIENT_FAILURE_RESPONSE_CODES,
)


_logger = logging.getLogger(__name__)
_gateway_uri: Optional[str] = None


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, hyphen, and dot.

    Returns True if the string doesn't contain any of these characters.
    """
    return bool(re.fullmatch(r"[\w\-\.]+", name))


def check_configuration_route_name_collisions(config):
    if len(config["routes"]) < 2:
        return
    names = [route["name"] for route in config["routes"]]
    if len(names) != len(set(names)):
        raise MlflowException.invalid_parameter_value(
            "Duplicate names found in route configurations. Please remove the duplicate route "
            "name from the configuration to ensure that route endpoints are created properly."
        )


def kill_child_processes(parent_pid):
    """
    Gracefully terminate or kill child processes from a main process
    """
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.terminate()
    _, still_alive = psutil.wait_procs(parent.children(), timeout=3)
    for p in still_alive:
        p.kill()


def _is_valid_uri(uri: str):
    """
    Evaluates the basic structure of a provided gateway uri to determine if the scheme and
    netloc are provided
    """
    try:
        parsed = urlparse(uri)
        return all([parsed.scheme, parsed.netloc])
    except ValueError:
        return False


def _is_gateway_server_available(gateway_uri: str):
    server_health_url = urljoin(gateway_uri, MLFLOW_GATEWAY_HEALTH_ENDPOINT)

    try:
        response = _get_gateway_response_with_retries("GET", server_health_url)
        augmented_raise_for_status(response)
    except HTTPError as http_err:
        _logger.warning(f"There is not a gateway server running at {gateway_uri}. {http_err}")
        return False
    except Exception as err:
        _logger.warning(
            f"Unable to verify if a gateway server is healthy at {gateway_uri}. Error: {err}"
        )
        return False
    else:
        return True


def _merge_uri_paths(paths: List[str]) -> str:
    sanitized = [part.strip("/") for part in paths]
    return "/".join(sanitized)


def _resolve_gateway_uri(gateway_uri: Optional[str] = None) -> str:
    gateway = gateway_uri or get_gateway_uri()
    return gateway


def set_gateway_uri(gateway_uri: str):
    if not _is_valid_uri(gateway_uri):
        raise MlflowException.invalid_parameter_value(
            "The gateway uri provided is missing required elements. Ensure that the schema "
            "and netloc are provided."
        )

    if not _is_gateway_server_available(gateway_uri):
        raise MlflowException.invalid_parameter_value(
            f"The gateway server cannot be verified at {gateway_uri}. Please verify that the "
            "server has been started and that you are able to ping it."
        )
    global _gateway_uri
    _gateway_uri = gateway_uri


def get_gateway_uri() -> str:
    global _gateway_uri
    if _gateway_uri is not None:
        return _gateway_uri
    elif (uri := env.get_env(MLFLOW_GATEWAY_URI.name)) is not None:
        return uri
    else:
        raise MlflowException(
            "No Gateway server uri has been set. Please either set the MLflow Gateway URI via "
            f"the fluent API or set the environment variable {MLFLOW_GATEWAY_URI.name} to the "
            "running Gateway API server's uri"
        )


def _validate_gateway_uri_is_set(func):
    """
    Validates that the MLflow Gateway server uri has been set
    """

    @wraps(func)
    def function(*args, **kwargs):
        get_gateway_uri()
        return func(*args, **kwargs)

    return function


def _get_gateway_response_with_retries(method, url, **kwargs):
    return _get_http_response_with_retries(
        method=method,
        url=url,
        max_retries=MLFLOW_HTTP_REQUEST_MAX_RETRIES.get(),
        backoff_factor=MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.get(),
        retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES,
        **kwargs,
    )
