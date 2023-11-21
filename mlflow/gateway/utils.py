import base64
import functools
import inspect
import json
import logging
import re
import textwrap
import warnings
from typing import Optional

from mlflow.deployments.utils import (
    _is_valid_uri,  # noqa: F401
    assemble_uri_path,  # noqa: F401
    resolve_route_url,  # noqa: F401
)
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES

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
    import psutil

    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, still_alive = psutil.wait_procs(parent.children(), timeout=3)
    for p in still_alive:
        p.kill()


def _get_indent(s: str) -> str:
    for l in s.splitlines():
        if l.startswith(" "):
            return " " * (len(l) - len(l.lstrip()))
    return ""


def _prepend(docstring: Optional[str], text: str) -> str:
    if not docstring:
        return text

    indent = _get_indent(docstring)
    return f"""
{textwrap.indent(text, indent)}

{docstring}
"""


def gateway_deprecated(obj):
    msg = (
        "MLflow AI gateway is deprecated and has been replaced by the deployments API for "
        "generative AI. See https://mlflow.org/docs/latest/llms/gateway/deprecation.html for "
        "more details."
    )
    warning = f"""
.. warning::

    {msg}
""".strip()
    if inspect.isclass(obj):
        original = obj.__init__

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return original(*args, **kwargs)

        obj.__init__ = wrapper
        obj.__init__.__doc__ = _prepend(obj.__init__.__doc__, warning)
        return obj
    else:

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__doc__ = _prepend(obj.__doc__, warning)

        return wrapper


@gateway_deprecated
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


@gateway_deprecated
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


def is_valid_ai21labs_model(model_name: str) -> bool:
    return model_name in {"j2-ultra", "j2-mid", "j2-light"}
