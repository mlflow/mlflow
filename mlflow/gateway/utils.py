import base64
import json
import logging
import posixpath
import re
from typing import List, Optional
from urllib.parse import urlparse

import psutil

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import append_to_uri_path

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
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, still_alive = psutil.wait_procs(parent.children(), timeout=3)
    for p in still_alive:
        p.kill()


def _is_valid_uri(uri: str):
    """
    Evaluates the basic structure of a provided gateway uri to determine if the scheme and
    netloc are provided
    """
    if uri == "databricks":
        return True
    try:
        parsed = urlparse(uri)
        return parsed.scheme == "databricks" or all([parsed.scheme, parsed.netloc])
    except ValueError:
        return False


@experimental
def set_gateway_uri(gateway_uri: str):
    """
    Sets the uri of a configured and running MLflow AI Gateway server in a global context.
    Providing a valid uri and calling this function is required in order to use the MLflow
    AI Gateway fluent APIs.

    :param gateway_uri: The full uri of a running MLflow AI Gateway server or, if running on
                        Databricks, "databricks".
    """
    if not _is_valid_uri(gateway_uri):
        raise MlflowException.invalid_parameter_value(
            "The gateway uri provided is missing required elements. Ensure that the schema "
            "and netloc are provided."
        )

    global _gateway_uri
    _gateway_uri = gateway_uri


@experimental
def get_gateway_uri() -> str:
    """
    Returns the currently set MLflow AI Gateway server uri iff set.
    If the Gateway uri has not been set by using ``set_gateway_uri``, an ``MlflowException``
    is raised.
    """
    global _gateway_uri
    if _gateway_uri is not None:
        return _gateway_uri
    elif uri := MLFLOW_GATEWAY_URI.get():
        return uri
    else:
        raise MlflowException(
            "No Gateway server uri has been set. Please either set the MLflow Gateway URI via "
            "`mlflow.gateway.set_gateway_uri()` or set the environment variable "
            f"{MLFLOW_GATEWAY_URI} to the running Gateway API server's uri"
        )


def assemble_uri_path(paths: List[str]) -> str:
    """
    Assemble a correct URI path from a list of path parts.

    :param paths: A list of strings representing parts of a URI path.
    :return: A string representing the complete assembled URI path.
    """
    stripped_paths = [path.strip("/").lstrip("/") for path in paths if path]
    return "/" + posixpath.join(*stripped_paths) if stripped_paths else "/"


def resolve_route_url(base_url: str, route: str) -> str:
    """
    Performs a validation on whether the returned value is a fully qualified url (as the case
    with Databricks) or requires the assembly of a fully qualified url by appending the
    Route return route_url to the base url of the AI Gateway server.

    :param base_url: The base URL. Should include the scheme and domain, e.g.,
                     ``http://127.0.0.1:6000``.
    :param route: The route to be appended to the base URL, e.g., ``/api/2.0/gateway/routes/`` or,
                  in the case of Databricks, the fully qualified url.
    :return: The complete URL, either directly returned or formed and returned by joining the
             base URL and the route path.
    """
    return route if _is_valid_uri(route) else append_to_uri_path(base_url, route)


class SearchRoutesToken:
    def __init__(self, index: int):
        self._index = index

    @property
    def index(self):
        return self._index

    @classmethod
    def decode(cls, encoded_token: str):
        try:
            decoded_token = base64.b64decode(encoded_token)
            parsed_token = json.loads(decoded_token)
            index = int(parsed_token.get("index"))
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Invalid SearchRoutes token: {encoded_token}. The index is not defined as a "
                "value that can be represented as a positive integer."
            ) from e

        if index < 0:
            raise MlflowException.invalid_parameter_value(
                f"Invalid SearchRoutes token: {encoded_token}. The index cannot be negative."
            )

        return cls(index=index)

    def encode(self) -> str:
        token_json = json.dumps(
            {
                "index": self.index,
            }
        )
        encoded_token_bytes = base64.b64encode(bytes(token_json, "utf-8"))
        return encoded_token_bytes.decode("utf-8")


def is_valid_mosiacml_chat_model(model_name: str) -> bool:
    return any(
        model_name.lower().startswith(supported)
        for supported in MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
    )
