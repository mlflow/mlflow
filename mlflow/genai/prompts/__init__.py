import warnings
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

import mlflow.tracking._model_registry.fluent as registry_api
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.prompt.registry_utils import require_prompt_registry
from mlflow.store.entities.paged_list import PagedList
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


@experimental(version="3.0.0")
@require_prompt_registry
def register_prompt(
    name: str,
    template: str | list[dict[str, Any]],
    commit_message: str | None = None,
    tags: dict[str, str] | None = None,
    response_format: BaseModel | dict[str, Any] | None = None,
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
        )


@experimental(version="3.0.0")
@require_prompt_registry
def search_prompts(
    filter_string: str | None = None,
    max_results: int | None = None,
) -> PagedList[Prompt]:
    with suppress_genai_migration_warning():
        return registry_api.search_prompts(filter_string=filter_string, max_results=max_results)


@experimental(version="3.0.0")
@require_prompt_registry
def load_prompt(
    name_or_uri: str,
    version: str | int | None = None,
    allow_missing: bool = False,
) -> PromptVersion:
    """
    Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

    The prompt can be specified by name and version, or by URI.

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt (required when using name, not allowed when using URI).
        allow_missing: If True, return None instead of raising Exception if the specified prompt
            is not found.

    Example:

    .. code-block:: python

        import mlflow

        # Load a specific version of the prompt
        prompt = mlflow.genai.load_prompt("my_prompt", version=1)

        # Load a specific version of the prompt by URI
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt/1")

        # Load a prompt version with an alias "production"
        prompt = mlflow.genai.load_prompt("prompts:/my_prompt@production")

    """
    with suppress_genai_migration_warning():
        return registry_api.load_prompt(
            name_or_uri=name_or_uri, version=version, allow_missing=allow_missing
        )


@experimental(version="3.0.0")
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


@experimental(version="3.0.0")
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
