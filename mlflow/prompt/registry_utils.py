import functools
import re
from textwrap import dedent
from typing import Any, Optional, Union

import mlflow
from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY, PROMPT_NAME_RULE
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS


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


def has_prompt_tag(tags: Optional[Union[list[RegisteredModelTag], dict[str, str]]]) -> bool:
    """Check if the given tags contain the prompt tag."""
    if isinstance(tags, dict):
        return IS_PROMPT_TAG_KEY in tags if tags else False
    if not tags:
        return
    return any(tag.key == IS_PROMPT_TAG_KEY for tag in tags)


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


def validate_prompt_name(name: Any):
    """Validate the prompt name against the prompt specific rule"""
    if not isinstance(name, str) or not name:
        raise MlflowException.invalid_parameter_value(
            "Prompt name must be a non-empty string.",
        )

    if PROMPT_NAME_RULE.match(name) is None:
        raise MlflowException.invalid_parameter_value(
            "Prompt name can only contain alphanumeric characters, hyphens, underscores, and dots.",
        )


def handle_resource_already_exist_error(
    name: str,
    is_existing_entity_prompt: bool,
    is_new_entity_prompt: bool,
):
    """
    Show a more specific error message for name conflict in Model Registry.

    1. When creating a model with the same name as an existing model, say "model already exists".
    2. When creating a prompt with the same name as an existing prompt, say "prompt already exists".
    3. Otherwise, explain that a prompt and a model cannot have the same name.
    """
    old_entity = "Prompt" if is_existing_entity_prompt else "Registered Model"
    new_entity = "Prompt" if is_new_entity_prompt else "Registered Model"

    if old_entity != new_entity:
        raise MlflowException(
            f"Tried to create a {new_entity.lower()} with name {name!r}, but the name is "
            f"already taken by a {old_entity.lower()}. MLflow does not allow creating a "
            "model and a prompt with the same name.",
            RESOURCE_ALREADY_EXISTS,
        )

    raise MlflowException(
        f"{new_entity} (name={name}) already exists.",
        RESOURCE_ALREADY_EXISTS,
    )
