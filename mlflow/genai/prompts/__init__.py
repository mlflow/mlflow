import json
import warnings
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

import mlflow.tracking._model_registry.fluent as registry_api
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import (
    PromptModelConfig,
    PromptVersion,
)
from mlflow.prompt.constants import PROMPT_MODEL_CONFIG_TAG_KEY
from mlflow.prompt.registry_utils import PromptCache as PromptCache
from mlflow.prompt.registry_utils import require_prompt_registry
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental


@contextmanager
def suppress_genai_migration_warning():
    """Suppress the deprecation warning when the api is called from `mlflow.genai` namespace."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=FutureWarning,
            message="The `mlflow.*` API is moved to the `mlflow.genai` namespace.*",
        )
        yield


@require_prompt_registry
def register_prompt(
    name: str,
    template: str | list[dict[str, Any]],
    commit_message: str | None = None,
    tags: dict[str, str] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    model_config: PromptModelConfig | dict[str, Any] | None = None,
) -> PromptVersion:
    """
    Register a new :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    A :py:class:`Prompt <mlflow.entities.Prompt>` is a pair of name and
    template content at minimum. With MLflow Prompt Registry, you can create, manage, and
    version control prompts with the MLflow's robust model tracking framework.

    If there is no registered prompt with the given name, a new prompt will be created.
    Otherwise, a new version of the existing prompt will be created.

    Args:
        name: The name of the prompt.
        template: The template content of the prompt. Can be either:

            - A string containing text with variables enclosed in double curly braces,
              e.g. {{variable}}, which will be replaced with actual values by the `format` method.
            - A list of dictionaries representing chat messages, where each message has
              'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello {{name}}"}])


            .. note::

                If you want to use the prompt with a framework that uses single curly braces
                e.g. LangChain, you can use the `to_single_brace_format` method to convert the
                loaded prompt to a format that uses single curly braces.

                .. code-block:: python

                    prompt = client.load_prompt("my_prompt")
                    langchain_format = prompt.to_single_brace_format()

        commit_message: A message describing the changes made to the prompt, similar to a
            Git commit message. Optional.
        tags: A dictionary of tags associated with the **prompt version**.
            This is useful for storing version-specific information, such as the author of
            the changes. Optional.
        response_format: Optional Pydantic class or dictionary defining the expected response
            structure. This can be used to specify the schema for structured outputs from LLM calls.
        model_config: Optional PromptModelConfig instance or dictionary containing model-specific
            configuration including model name and settings like temperature, top_p, max_tokens.
            Using PromptModelConfig provides validation and type safety for common parameters.
            Example (dict): {"model_name": "gpt-4", "temperature": 0.7}
            Example (PromptModelConfig): PromptModelConfig(model_name="gpt-4", temperature=0.7)

    Returns:
        A :py:class:`Prompt <mlflow.entities.Prompt>` object that was created.

    Example:

    .. code-block:: python

        import mlflow

        # Register a text prompt
        mlflow.genai.register_prompt(
            name="greeting_prompt",
            template="Respond to the user's message as a {{style}} AI.",
        )

        # Register a chat prompt with multiple messages
        mlflow.genai.register_prompt(
            name="assistant_prompt",
            template=[
                {"role": "system", "content": "You are a helpful {{style}} assistant."},
                {"role": "user", "content": "{{question}}"},
            ],
            response_format={"type": "object", "properties": {"answer": {"type": "string"}}},
        )

        # Load and use the prompt
        prompt = mlflow.genai.load_prompt("greeting_prompt")

        # Use the prompt in your application
        import openai

        openai_client = openai.OpenAI()
        openai_client.chat.completion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt.format(style="friendly")},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )

        # Update the prompt with a new version
        prompt = mlflow.genai.register_prompt(
            name="greeting_prompt",
            template="Respond to the user's message as a {{style}} AI. {{greeting}}",
            commit_message="Add a greeting to the prompt.",
            tags={"author": "Bob"},
        )

    """
    with suppress_genai_migration_warning():
        return registry_api.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags,
            response_format=response_format,
            model_config=model_config,
        )


@require_prompt_registry
def search_prompts(
    filter_string: str | None = None,
    max_results: int | None = None,
) -> PagedList[Prompt]:
    with suppress_genai_migration_warning():
        return registry_api.search_prompts(filter_string=filter_string, max_results=max_results)


@require_prompt_registry
def load_prompt(
    name_or_uri: str,
    version: str | int | None = None,
    allow_missing: bool = False,
    link_to_model: bool = True,
    model_id: str | None = None,
    cache_ttl_seconds: float | None = None,
) -> PromptVersion:
    """
    Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

    The prompt can be specified by name and version, or by URI.

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt (required when using name, not allowed when using URI).
        allow_missing: If True, return None instead of raising Exception if the specified prompt
            is not found.
        link_to_model: If True, link the prompt to the model.
        model_id: The ID of the model to link the prompt to. Only used if link_to_model is True.
        cache_ttl_seconds: Time-to-live in seconds for the cached prompt. If not specified,
            uses the value from `MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS` environment variable for
            alias-based prompts (default 60), and the value from
            `MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS` environment variable for version-based prompts
            (default None, no TTL).
            Set to 0 to bypass the cache and always fetch from the server.

    Example:

    .. code-block:: python

        import mlflow

        # Load the latest version of the prompt
        prompt = mlflow.genai.load_prompt("my_prompt")

        # Load a specific version of the prompt
        prompt = mlflow.genai.load_prompt("my_prompt", version=1)

        # Load a specific version of the prompt by URI
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt/1")

        # Load a prompt version with an alias "production"
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt@production")

        # Load the latest version of the prompt by URI
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt@latest")

        # Load with custom cache TTL (5 minutes)
        prompt = mlflow.genai.load_prompt("my_prompt", version=1, cache_ttl_seconds=300)

        # Bypass cache entirely
        prompt = mlflow.genai.load_prompt("my_prompt", version=1, cache_ttl_seconds=0)
    """
    with suppress_genai_migration_warning():
        return registry_api.load_prompt(
            name_or_uri=name_or_uri,
            version=version,
            allow_missing=allow_missing,
            link_to_model=link_to_model,
            model_id=model_id,
            cache_ttl_seconds=cache_ttl_seconds,
        )


@require_prompt_registry
def set_prompt_alias(name: str, alias: str, version: int) -> None:
    """
    Set an alias for a :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        alias: The alias to set for the prompt.
        version: The version of the prompt.

    Example:

    .. code-block:: python

        import mlflow

        # Set an alias for the prompt
        mlflow.genai.set_prompt_alias(name="my_prompt", version=1, alias="production")

        # Load the prompt by alias (use "@" to specify the alias)
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt@production")

        # Switch the alias to a new version of the prompt
        mlflow.genai.set_prompt_alias(name="my_prompt", version=2, alias="production")

        # Delete the alias
        mlflow.genai.delete_prompt_alias(name="my_prompt", alias="production")
    """
    with suppress_genai_migration_warning():
        return registry_api.set_prompt_alias(name=name, version=version, alias=alias)


@require_prompt_registry
def delete_prompt_alias(name: str, alias: str) -> None:
    """
    Delete an alias for a :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        alias: The alias to delete for the prompt.
    """
    with suppress_genai_migration_warning():
        return registry_api.delete_prompt_alias(name=name, alias=alias)


@experimental(version="3.5.0")
@require_prompt_registry
def get_prompt_tags(name: str) -> Prompt:
    """Get a prompt's metadata from the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
    """
    with suppress_genai_migration_warning():
        return MlflowClient().get_prompt(name=name).tags


@experimental(version="3.5.0")
@require_prompt_registry
def set_prompt_tag(name: str, key: str, value: str) -> None:
    """Set a tag on a prompt in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        key: The key of the tag
        value: The value of the tag for the key
    """
    with suppress_genai_migration_warning():
        MlflowClient().set_prompt_tag(name=name, key=key, value=value)


@experimental(version="3.5.0")
@require_prompt_registry
def delete_prompt_tag(name: str, key: str) -> None:
    """Delete a tag from a prompt in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        key: The key of the tag
    """
    with suppress_genai_migration_warning():
        MlflowClient().delete_prompt_tag(name=name, key=key)


@experimental(version="3.5.0")
@require_prompt_registry
def set_prompt_version_tag(name: str, version: str | int, key: str, value: str) -> None:
    """Set a tag on a prompt version in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        version: The version of the prompt.
        key: The key of the tag
        value: The value of the tag for the key
    """
    with suppress_genai_migration_warning():
        MlflowClient().set_prompt_version_tag(name=name, version=version, key=key, value=value)


@experimental(version="3.5.0")
@require_prompt_registry
def delete_prompt_version_tag(name: str, version: str | int, key: str) -> None:
    """Delete a tag from a prompt version in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        version: The version of the prompt.
        key: The key of the tag
    """
    with suppress_genai_migration_warning():
        MlflowClient().delete_prompt_version_tag(name=name, version=version, key=key)


@experimental(version="3.8.0")
@require_prompt_registry
def set_prompt_model_config(
    name: str,
    version: str | int,
    model_config: PromptModelConfig | dict[str, Any],
) -> None:
    """Set or update the model configuration for a specific prompt version.

    Model configuration includes model-specific settings such as model name, temperature,
    max_tokens, and other inference parameters. Unlike the prompt template, model configuration
    is mutable and can be updated after a prompt version is created.

    Args:
        name: The name of the prompt.
        version: The version of the prompt.
        model_config: A PromptModelConfig or dict with model settings like model_name, temperature.

    Example:

    .. code-block:: python

        import mlflow
        from mlflow.entities.model_registry import PromptModelConfig

        # Set model config using a dictionary
        mlflow.genai.set_prompt_model_config(
            name="my-prompt",
            version=1,
            model_config={"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
        )

        # Set model config using PromptModelConfig for validation
        config = PromptModelConfig(
            model_name="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2000,
            top_p=0.95,
        )
        mlflow.genai.set_prompt_model_config(
            name="my-prompt",
            version=1,
            model_config=config,
        )

        # Load and verify the config was set
        prompt = mlflow.genai.load_prompt("my-prompt", version=1)
        print(prompt.model_config)
    """
    if isinstance(model_config, PromptModelConfig):
        config_dict = model_config.to_dict()
    else:
        config_dict = PromptModelConfig.from_dict(model_config).to_dict()

    config_json = json.dumps(config_dict)

    with suppress_genai_migration_warning():
        MlflowClient().set_prompt_version_tag(
            name=name, version=version, key=PROMPT_MODEL_CONFIG_TAG_KEY, value=config_json
        )


@experimental(version="3.8.0")
@require_prompt_registry
def delete_prompt_model_config(name: str, version: str | int) -> None:
    """Delete the model configuration from a specific prompt version.

    Args:
        name: The name of the prompt.
        version: The version of the prompt.

    Example:

    .. code-block:: python

        import mlflow

        # Remove model config from a prompt version
        mlflow.genai.delete_prompt_model_config(name="my-prompt", version=1)

        # Verify the config was removed
        prompt = mlflow.genai.load_prompt("my-prompt", version=1)
        assert prompt.model_config is None
    """
    with suppress_genai_migration_warning():
        MlflowClient().delete_prompt_version_tag(
            name=name, version=version, key=PROMPT_MODEL_CONFIG_TAG_KEY
        )
