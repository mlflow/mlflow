import functools
import urllib.parse
from textwrap import dedent
from typing import Optional

import mlflow
from mlflow.entities.model_registry.prompt import IS_PROMPT_TAG_KEY
from mlflow.exceptions import MlflowException


def add_prompt_filter_string(
    filter_string: Optional[str], is_prompt: bool = False
) -> Optional[str]:
    """
    Additional filter string to include/exclude prompts from the result.
    By default, exclude prompts from the result.
    """
    if IS_PROMPT_TAG_KEY not in (filter_string or ""):
        prompt_filter_query = (
            f"tag.`{IS_PROMPT_TAG_KEY}` = 'true'"
            if is_prompt
            else f"tag.`{IS_PROMPT_TAG_KEY}` != 'true'"
        )
        if filter_string:
            filter_string = f"{filter_string} AND {prompt_filter_query}"
        else:
            filter_string = prompt_filter_query
    return filter_string


def parse_prompt_uri(uri: str) -> tuple[str, str]:
    """
    Parse prompt URI into prompt name and prompt version.
    'prompt:/<name>/<version>' -> ('<name>', '<version>')
    """
    parsed = urllib.parse.urlparse(uri)

    if parsed.scheme != "prompts":
        raise MlflowException.invalid_parameter_value(
            f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
        )

    path = parsed.path
    if path.count("/") != 2:
        raise MlflowException.invalid_parameter_value(
            f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
        )

    return path.split("/")[1:]


def has_prompt_tag(tags: Optional[dict]) -> bool:
    """Check if the given tags contain the prompt tag."""
    return IS_PROMPT_TAG_KEY in tags if tags else False


def is_prompt_supported_registry(registry_uri: Optional[str] = None) -> bool:
    """
    Check if the current registry supports prompts.

    Prompts registration is supported only in the OSS MLflow Tracking Server,
    not in Databricks or OSS Unity Catalog.
    """
    registry_uri = registry_uri or mlflow.get_registry_uri()
    return not registry_uri.startswith("databricks") and not registry_uri.startswith("uc:")


def require_prompt_registry(func):
    """Ensure that the current registry supports prompts."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], mlflow.MlflowClient):
            registry_uri = args[0]._registry_uri
        else:
            registry_uri = mlflow.get_registry_uri()

        if not is_prompt_supported_registry(registry_uri):
            raise MlflowException(
                f"The '{func.__name__}' API is not supported in the current model registry "
                "type. Prompts are supported only in the OSS MLflow Tracking Server."
            )
        return func(*args, **kwargs)

    # Add note about prompt support to the docstring
    func.__doc__ = dedent(f"""\
        {func.__doc__}

        .. note::

            This API is supported only when using the OSS MLflow Model Registry. Prompts are not
            supported in Databricks or the OSS Unity Catalog model registry.
    """)
    return wrapper
