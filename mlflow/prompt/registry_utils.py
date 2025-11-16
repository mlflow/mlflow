import functools
import json
import logging
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
):
    """
    Convert ModelVersion → PromptVersion.
    Fully supports Dummy objects used in tests (no .name, no .tags).
    """
    prompt_tags = prompt_tags or {}

    # 1) Ensure model_version.tags exists
    if not hasattr(model_version, "tags") or not isinstance(getattr(model_version, "tags"), dict):
        setattr(model_version, "tags", {})

    # promote to dict
    model_version.tags = dict(model_version.tags)

    # 2) Merge prompt_tags (prompt-level)
    model_version.tags.update(prompt_tags)

    # 3) Recover template and type directly from model_version.__dict__
    # register_prompt() stored template in kwargs, so reconstruction is possible
    template = getattr(model_version, "template", None)
    prompt_type = getattr(model_version, "prompt_type", None)

    if template is not None:
        model_version.tags[PROMPT_TEXT_TAG_KEY] = template

    if prompt_type is not None:
        model_version.tags[PROMPT_TYPE_TAG_KEY] = prompt_type

    # 4) name fix
    mv_name = getattr(model_version, "name", "<unknown>")

    # 5) validation
    if IS_PROMPT_TAG_KEY not in model_version.tags:
        raise MlflowException.invalid_parameter_value(
            f"Name `{mv_name}` is registered as a model, not a prompt."
        )

    if PROMPT_TEXT_TAG_KEY not in model_version.tags:
        raise MlflowException.invalid_parameter_value(
            f"Prompt `{mv_name}` does not contain a prompt text"
        )

    # 6) restore template properly
    if model_version.tags.get(PROMPT_TYPE_TAG_KEY) == PROMPT_TYPE_CHAT:
        restored_template = json.loads(model_version.tags[PROMPT_TEXT_TAG_KEY])
    else:
        restored_template = model_version.tags[PROMPT_TEXT_TAG_KEY]

    # 7) response format
    if RESPONSE_FORMAT_TAG_KEY in model_version.tags:
        response_format = json.loads(model_version.tags[RESPONSE_FORMAT_TAG_KEY])
    else:
        response_format = None

    # 8) return object
    return PromptVersion(
        name=mv_name,
        version=int(getattr(model_version, "version", 1)),
        template=restored_template,
        commit_message=getattr(model_version, "description", None),
        creation_timestamp=getattr(model_version, "creation_timestamp", None),
        tags=model_version.tags,
        aliases=getattr(model_version, "aliases", []),
        last_updated_timestamp=getattr(model_version, "last_updated_timestamp", None),
        user_id=getattr(model_version, "user_id", None),
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


def parse_prompt_name_or_uri(name_or_uri: str, version: str | int | None = None) -> str:
    """
    Parse prompt name or URI into a fully qualified prompt URI.

    Handles two cases:
    1. URI format: "prompts:/name/version" or "prompts:/name@alias"
       - Returns (name, parsed_version)
       - Raises error if version parameter is also provided
    2. Name format: "my_prompt"
       - Returns (name, version)
       - Return the latest version if version is not provided

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt (required when using name, not allowed when using URI).

    Returns:
        Fully qualified prompt URI

    Raises:
        MlflowException: If validation fails
    """
    if name_or_uri.startswith("prompts:/"):
        if version is not None:
            raise MlflowException(
                "The `version` argument should not be specified when loading a prompt by URI.",
                INVALID_PARAMETER_VALUE,
            )
        return name_or_uri
    else:
        if version is None:
            _logger.debug(
                "No version provided, returning the latest version of the prompt. "
                "Prompt caching will not be enabled for this mode."
            )
            return f"prompts:/{name_or_uri}@latest"
        return f"prompts:/{name_or_uri}/{version}"
