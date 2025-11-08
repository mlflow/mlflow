import base64
import functools
import inspect
import json
import logging
import posixpath
import re
import textwrap
import warnings
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
from mlflow.utils.uri import append_to_uri_path

_logger = logging.getLogger(__name__)
_gateway_uri: str | None = None


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, hyphen, and dot.

    Returns True if the string doesn't contain any of these characters.
    """
    return bool(re.fullmatch(r"[\w\-\.]+", name))


def check_configuration_route_name_collisions(config):
    endpoints = config.get("endpoints") or []
    routes = config.get("routes") or []

    endpoint_names = [endpoint["name"] for endpoint in endpoints]
    route_names = [route["name"] for route in routes]

    merged_names = endpoint_names + route_names
    if len(merged_names) != len(set(merged_names)):
        raise MlflowException.invalid_parameter_value(
            "Duplicate names found in endpoint / route configurations. "
            "Please remove the duplicate endpoint / route name "
            "from the configuration to ensure that endpoints / routes are created properly."
        )

    endpoint_config_dict = {endpoint["name"]: endpoint for endpoint in endpoints}

    for route in routes:
        route_name = route["name"]
        route_task_type = route["task_type"]

        traffic_percentage_sum = 0
        for destination in route.get("destinations"):
            dest_name = destination.get("name")
            dest_traffic_percentage = destination.get("traffic_percentage")
            traffic_percentage_sum += dest_traffic_percentage
            if dest_name not in endpoint_names:
                raise MlflowException.invalid_parameter_value(
                    f"The route destination name must be a endpoint name, "
                    f"but the route '{route_name}' has an invalid destination name '{dest_name}'."
                )

            dest_endpoint_type = endpoint_config_dict[dest_name].get("endpoint_type")
            if route_task_type != dest_endpoint_type:
                raise MlflowException.invalid_parameter_value(
                    f"The route destination endpoint types in the route '{route_name}' must have "
                    f"endpoint type '{route_task_type}' but got endpoint type "
                    f"'{dest_endpoint_type}'."
                )

            if not (0 <= dest_traffic_percentage <= 100):
                raise MlflowException.invalid_parameter_value(
                    "The route destination traffic percentage must between 0 and 100."
                )

        if traffic_percentage_sum != 100:
            raise MlflowException.invalid_parameter_value(
                "For each route configuration, the traffic percentage sum of destinations "
                f"must be 100, but got invalid configuration of route '{route_name}'."
            )


def check_configuration_deprecated_fields(config):
    endpoints = config.get("endpoints", [])
    for endpoint in endpoints:
        if "route_type" in endpoint:
            raise MlflowException.invalid_parameter_value(
                "The 'route_type' configuration key is not supported in the configuration file. "
                "Use 'endpoint_type' instead."
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


def _get_indent(s: str) -> str:
    for l in s.splitlines():
        if l.startswith(" "):
            return " " * (len(l) - len(l.lstrip()))
    return ""


def _prepend(docstring: str | None, text: str) -> str:
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
        "generative AI. See https://mlflow.org/docs/latest/llms/gateway/migration.html for "
        "migration."
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
    """Sets the uri of a configured and running MLflow AI Gateway server in a global context.
    Providing a valid uri and calling this function is required in order to use the MLflow
    AI Gateway fluent APIs.

    Args:
        gateway_uri: The full uri of a running MLflow AI Gateway server or, if running on
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


def assemble_uri_path(paths: list[str]) -> str:
    """Assemble a correct URI path from a list of path parts.

    Args:
        paths: A list of strings representing parts of a URI path.

    Returns:
        A string representing the complete assembled URI path.

    """
    stripped_paths = [path.strip("/").lstrip("/") for path in paths if path]
    return "/" + posixpath.join(*stripped_paths) if stripped_paths else "/"


def resolve_route_url(base_url: str, route: str) -> str:
    """
    Performs a validation on whether the returned value is a fully qualified url (as the case
    with Databricks) or requires the assembly of a fully qualified url by appending the
    Route return route_url to the base url of the AI Gateway server.

    Args:
        base_url: The base URL. Should include the scheme and domain, e.g.,
            ``http://127.0.0.1:6000``.
        route: The route to be appended to the base URL, e.g., ``/api/2.0/gateway/routes/`` or,
            in the case of Databricks, the fully qualified url.

    Returns:
        The complete URL, either directly returned or formed and returned by joining the
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


def is_valid_ai21labs_model(model_name: str) -> bool:
    return model_name in {"j2-ultra", "j2-mid", "j2-light"}


def strip_sse_prefix(s: str) -> str:
    # https://html.spec.whatwg.org/multipage/server-sent-events.html
    return re.sub(r"^data:\s+", "", s)


def to_sse_chunk(data: str) -> str:
    # https://html.spec.whatwg.org/multipage/server-sent-events.html
    return f"data: {data}\n\n"


def _find_boundary(buffer: bytes) -> int:
    try:
        return buffer.index(b"\n")
    except ValueError:
        return -1


async def handle_incomplete_chunks(
    stream: AsyncGenerator[bytes, Any],
) -> AsyncGenerator[bytes, Any]:
    """
    Wraps a streaming response and handles incomplete chunks from the server.
    See https://community.openai.com/t/incomplete-stream-chunks-for-completions-api/383520
    for more information.
    """
    buffer = b""
    async for chunk in stream:
        buffer += chunk
        while (boundary := _find_boundary(buffer)) != -1:
            yield buffer[:boundary]
            buffer = buffer[boundary + 1 :]


async def make_streaming_response(resp):
    from starlette.responses import StreamingResponse

    if isinstance(resp, AsyncGenerator):
        return StreamingResponse(
            (to_sse_chunk(d.json()) async for d in resp),
            media_type="text/event-stream",
        )
    else:
        return await resp
