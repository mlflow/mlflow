import functools
import re
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
                f"The '{func.__name__}' API is only available with the OSS MLflow Tracking Server."
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


def translate_prompt_exception(func):
    """
    Translate MlflowException message related to RegisteredModel / ModelVersion into
    prompt-specific message.
    """
    MODEL_PATTERN = re.compile(r"(registered model|model version)", re.IGNORECASE)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MlflowException as e:
            original_message = e.message
            # Preserve the case of the first letter
            new_message = MODEL_PATTERN.sub(
                lambda m: "Prompt" if m.group(0)[0].isupper() else "prompt", e.message
            )

            if new_message != original_message:
                raise MlflowException(new_message) from e
            else:
                raise e

    return wrapper
