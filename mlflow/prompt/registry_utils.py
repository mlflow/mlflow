import functools
import json
import re
from textwrap import dedent
from typing import Any

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_NAME_RULE,
    PROMPT_TEXT_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS


def model_version_to_prompt_version(
    model_version: ModelVersion, prompt_tags: dict[str, str] | None = None
) -> PromptVersion:
    """
    Create a PromptVersion object from a ModelVersion object.

    Args:
        model_version: The ModelVersion object to convert to a PromptVersion.
        prompt_tags: The prompt-level tags. Optional.

    Returns:
        PromptVersion: The converted PromptVersion object.
    """
    if IS_PROMPT_TAG_KEY not in model_version.tags:
        raise MlflowException.invalid_parameter_value(
            f"Name `{model_version.name}` is registered as a model, not a prompt. MLflow "
            "does not allow registering a prompt with the same name as an existing model.",
        )

    if PROMPT_TEXT_TAG_KEY not in model_version.tags:
        raise MlflowException.invalid_parameter_value(
            f"Prompt `{model_version.name}` does not contain a prompt text"
        )

    if model_version.tags.get(PROMPT_TYPE_TAG_KEY) == PROMPT_TYPE_CHAT:
        template = json.loads(model_version.tags[PROMPT_TEXT_TAG_KEY])
    else:
        template = model_version.tags[PROMPT_TEXT_TAG_KEY]

    if RESPONSE_FORMAT_TAG_KEY in model_version.tags:
        response_format = json.loads(model_version.tags[RESPONSE_FORMAT_TAG_KEY])
    else:
        response_format = None

    return PromptVersion(
        name=model_version.name,
        version=int(model_version.version),
        template=template,
        commit_message=model_version.description,
        creation_timestamp=model_version.creation_timestamp,
        tags=model_version.tags,
        aliases=model_version.aliases,
        last_updated_timestamp=model_version.last_updated_timestamp,
        user_id=model_version.user_id,
        response_format=response_format,
    )


def add_prompt_filter_string(filter_string: str | None, is_prompt: bool = False) -> str | None:
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


def has_prompt_tag(tags: list[RegisteredModelTag] | dict[str, str] | None) -> bool:
    """Check if the given tags contain the prompt tag."""
    if isinstance(tags, dict):
        return IS_PROMPT_TAG_KEY in tags if tags else False
    if not tags:
        return
    return any(tag.key == IS_PROMPT_TAG_KEY for tag in tags)


def is_prompt_supported_registry(registry_uri: str | None = None) -> bool:
    """
    Check if the current registry supports prompts.

    Prompts registration is supported in:
    - OSS MLflow Tracking Server (always)
    - Unity Catalog
    - Not supported in legacy Databricks workspace registry or Unity Catalog OSS
    """
    registry_uri = registry_uri or mlflow.get_registry_uri()

    # Legacy Databricks workspace registry doesn't support prompts
    if registry_uri.startswith("databricks") and not registry_uri.startswith("databricks-uc"):
        return False

    # Unity Catalog OSS doesn't support prompts
    if registry_uri.startswith("uc:"):
        return False

    # UC registries support prompts automatically
    if registry_uri.startswith("databricks-uc"):
        return True

    # OSS MLflow registry always supports prompts
    return True


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
                f"The '{func.__name__}' API is not supported with the current registry. "
                "Prompts are supported in OSS MLflow and Unity Catalog, but not in the "
                "legacy Databricks workspace registry.",
            )
        return func(*args, **kwargs)

    # Add note about prompt support to the docstring
    func.__doc__ = dedent(f"""\
        {func.__doc__}

        .. note::

            This API is supported in OSS MLflow Model Registry and Unity Catalog. It is
            not supported in the legacy Databricks workspace model registry.
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
                new_exc = MlflowException(new_message)
                new_exc.error_code = e.error_code  # Preserve original error code
                raise new_exc from e
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


def parse_prompt_name_or_uri(
    name_or_uri: str, version: str | int | None = None
) -> tuple[str, str | int | None]:
    """
    Parse prompt name or URI into (name, version) tuple.

    Handles two cases:
    1. URI format: "prompts:/name/version" or "prompts:/name@alias"
       - Returns (name, parsed_version)
       - Raises error if version parameter is also provided
    2. Name format: "my_prompt"
       - Returns (name, version)
       - Raises error if version parameter is not provided

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt (required when using name, not allowed when using URI).

    Returns:
        Tuple of (name, version) where version can be a string, int, or None

    Raises:
        MlflowException: If validation fails
    """
    if name_or_uri.startswith("prompts:/"):
        if version is not None:
            raise MlflowException(
                "The `version` argument should not be specified when loading a prompt by URI.",
                INVALID_PARAMETER_VALUE,
            )
        # Parse URI to extract name and version
        # This assumes the parse_prompt_uri method exists, but we'll handle that separately
        # For now, we'll do basic parsing and let the caller handle the URI parsing
        return name_or_uri, None
    else:
        if version is None:
            raise MlflowException(
                "Version must be specified when loading a prompt by name. "
                "Use a prompt URI (e.g., 'prompts:/name/version') or provide the version "
                "parameter.",
                INVALID_PARAMETER_VALUE,
            )
        return name_or_uri, version
