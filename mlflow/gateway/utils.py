from functools import wraps
import logging
import psutil
import re
import requests
from requests import HTTPError
from typing import Optional, List
from urllib.parse import urlparse, urljoin

from mlflow.exceptions import MlflowException
from mlflow.gateway.envs import MLFLOW_GATEWAY_URI  # TODO: change to environment_variables import
from mlflow.gateway.constants import MLFLOW_GATEWAY_HEALTH_ENDPOINT
from mlflow.utils.request_utils import augmented_raise_for_status


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
    try:
        health_endpoint = urljoin(gateway_uri, MLFLOW_GATEWAY_HEALTH_ENDPOINT)
        response = requests.get(health_endpoint)
        augmented_raise_for_status(response)
        return True
    except HTTPError as http_err:
        _logger.warning(f"There is not a gateway server running at {gateway_uri}. {http_err}")
    except Exception as err:
        _logger.warning(
            f"Unable to verify if a gateway server is healthy at {gateway_uri}. Error: {err}"
        )
    return False


def _merge_uri_paths(paths: List[str]) -> str:
    sanitized = [part.strip("/") for part in paths]
    merged = "/".join(sanitized)
    if not merged.startswith("/"):
        merged = "/" + merged
    return merged


def _resolve_gateway_uri(gateway_uri: Optional[str] = None) -> str:
    return gateway_uri or get_gateway_uri()


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
    elif uri := MLFLOW_GATEWAY_URI.get():
        return uri
    else:
        raise MlflowException(
            "No Gateway server uri has been set. Please either set the MLflow Gateway URI via "
            f"`mlflow.set_gateway_uri()` or set the environment variable {MLFLOW_GATEWAY_URI.name} "
            "to the running Gateway API server's uri"
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
