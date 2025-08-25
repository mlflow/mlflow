from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_TEMPLATE_VARIABLE_PATTERN,
    PROMPT_TEXT_DISPLAY_LIMIT,
    PROMPT_TEXT_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)

# Alias type
PromptVersionTag = ModelVersionTag


def _is_reserved_tag(key: str) -> bool:
    return key in {
        IS_PROMPT_TAG_KEY,
        PROMPT_TEXT_TAG_KEY,
        PROMPT_TYPE_TAG_KEY,
        RESPONSE_FORMAT_TAG_KEY,
    }


class PromptVersion(_ModelRegistryEntity):
    """
    An entity representing a specific version of a prompt with its template content.

    Args:
        name: The name of the prompt.
        version: The version number of the prompt.
        template: The template content of the prompt. Can be either:

            - A string containing text with variables enclosed in double curly braces,
              e.g. {{variable}}, which will be replaced with actual values by the `format` method.
              MLflow uses the same variable naming rules as Jinja2:
              https://jinja.palletsprojects.com/en/stable/api/#notes-on-identifiers
            - A list of dictionaries representing chat messages, where each message has
              'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello {{name}}"}])

        response_format: Optional Pydantic class or dictionary defining the expected response
            structure. This can be used to specify the schema for structured outputs.
        commit_message: The commit message for the prompt version. Optional.
        creation_timestamp: Timestamp of the prompt creation. Optional.
        tags: A dictionary of tags associated with the **prompt version**.
            This is useful for storing version-specific information, such as the author of
            the changes. Optional.
        aliases: List of aliases for this prompt version. Optional.
        last_updated_timestamp: Timestamp of last update. Optional.
        user_id: User ID that created this prompt version. Optional.

    """

    def __init__(
        self,
        name: str,
        version: int,
        template: str | list[dict[str, Any]],
        commit_message: str | None = None,
        creation_timestamp: int | None = None,
        tags: dict[str, str] | None = None,
        aliases: list[str] | None = None,
        last_updated_timestamp: int | None = None,
        user_id: str | None = None,
        response_format: BaseModel | dict[str, Any] | None = None,
    ):
        from mlflow.types.chat import ChatMessage

        super().__init__()

        # Core PromptVersion attributes
        self._name: str = name
        self._version: str = str(version)  # Store as string internally
        self._creation_time: int = creation_timestamp or 0

        # Initialize tags first
        tags = tags or {}

        # Determine prompt type and set it
        if isinstance(template, list) and len(template) > 0:
            try:
                for msg in template:
                    ChatMessage.model_validate(msg)
            except ValidationError as e:
                raise ValueError("Template must be a list of dicts with role and content") from e
            self._prompt_type = PROMPT_TYPE_CHAT
            tags[PROMPT_TYPE_TAG_KEY] = PROMPT_TYPE_CHAT
        else:
            self._prompt_type = PROMPT_TYPE_TEXT
            tags[PROMPT_TYPE_TAG_KEY] = PROMPT_TYPE_TEXT

        # Store template text as a tag
        tags[PROMPT_TEXT_TAG_KEY] = template if isinstance(template, str) else json.dumps(template)
        tags[IS_PROMPT_TAG_KEY] = "true"

        if response_format:
            tags[RESPONSE_FORMAT_TAG_KEY] = json.dumps(
                self.convert_response_format_to_dict(response_format)
            )

        # Store the tags dict
        self._tags: dict[str, str] = tags

        template_text = template if isinstance(template, str) else json.dumps(template)
        self._variables = set(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(template_text))
        self._last_updated_timestamp: int | None = last_updated_timestamp
        self._description: str | None = commit_message
        self._user_id: str | None = user_id
        self._aliases: list[str] = aliases or []

    def __repr__(self) -> str:
        if self.is_text_prompt:
            text = (
                self.template[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
                if len(self.template) > PROMPT_TEXT_DISPLAY_LIMIT
                else self.template
            )
        else:
            message = json.dumps(self.template)
            text = (
                message[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
                if len(message) > PROMPT_TEXT_DISPLAY_LIMIT
                else message
            )
        return f"PromptVersion(name={self.name}, version={self.version}, template={text})"

    # Core PromptVersion properties
    @property
    def template(self) -> str | list[dict[str, Any]]:
        """
        Return the template content of the prompt.

        Returns:
            Either a string (for text prompts) or a list of chat message dictionaries
            (for chat prompts) with 'role' and 'content' keys.
        """
        if self.is_text_prompt:
            return self._tags[PROMPT_TEXT_TAG_KEY]
        else:
            return json.loads(self._tags[PROMPT_TEXT_TAG_KEY])

    @property
    def is_text_prompt(self) -> bool:
        """
        Return True if the prompt is a text prompt, False if it's a chat prompt.

        Returns:
            True for text prompts (string templates), False for chat prompts (list of messages).
        """
        return self._prompt_type == PROMPT_TYPE_TEXT

    @property
    def response_format(self) -> dict[str, Any] | None:
        """
        Return the response format specification for the prompt.

        Returns:
            A dictionary defining the expected response structure, or None if no
            response format is specified. This can be used to validate or structure
            the output from LLM calls.
        """
        if RESPONSE_FORMAT_TAG_KEY not in self._tags:
            return None
        return json.loads(self._tags[RESPONSE_FORMAT_TAG_KEY])

    def to_single_brace_format(self) -> str | list[dict[str, Any]]:
        """
        Convert the template to single brace format. This is useful for integrating with other
        systems that use single curly braces for variable replacement, such as LangChain's prompt
        template.

        Returns:
            The template with variables converted from {{variable}} to {variable} format.
            For text prompts, returns a string. For chat prompts, returns a list of messages.
        """
        t = self.template if self.is_text_prompt else json.dumps(self.template)
        for var in self.variables:
            t = re.sub(r"\{\{\s*" + var + r"\s*\}\}", "{" + var + "}", t)
        return t if self.is_text_prompt else json.loads(t)

    @staticmethod
    def convert_response_format_to_dict(
        response_format: BaseModel | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert a response format specification to a dictionary representation.

        Args:
            response_format: Either a Pydantic BaseModel class or a dictionary defining
                the response structure.

        Returns:
            A dictionary representation of the response format. If a Pydantic class is
            provided, returns its JSON schema. If a dictionary is provided, returns it as-is.
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_json_schema()
        else:
            return response_format

    @property
    def variables(self) -> set[str]:
        """
        Return a list of variables in the template text.
        The value must be enclosed in double curly braces, e.g. {{variable}}.
        """
        return self._variables

    @property
    def commit_message(self) -> str | None:
        """
        Return the commit message of the prompt version.
        """
        return self.description

    @property
    def tags(self) -> dict[str, str]:
        """
        Return the version-level tags.
        """
        return {key: value for key, value in self._tags.items() if not _is_reserved_tag(key)}

    @property
    def uri(self) -> str:
        """Return the URI of the prompt."""
        return f"prompts:/{self.name}/{self.version}"

    @property
    def name(self) -> str:
        """String. Unique name within Model Registry."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def version(self) -> int:
        """Version"""
        return int(self._version)

    @property
    def creation_timestamp(self) -> int:
        """Integer. Prompt version creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self) -> int | None:
        """Integer. Timestamp of last update for this prompt version (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def description(self) -> str | None:
        """String. Description"""
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description

    @property
    def user_id(self) -> str | None:
        """String. User ID that created this prompt version."""
        return self._user_id

    @property
    def aliases(self) -> list[str]:
        """List of aliases (string) for the current prompt version."""
        return self._aliases

    @aliases.setter
    def aliases(self, aliases: list[str]):
        self._aliases = aliases

    # Methods
    @classmethod
    def _properties(cls) -> list[str]:
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag: ModelVersionTag):
        self._tags[tag.key] = tag.value

    def format(
        self, allow_partial: bool = False, **kwargs
    ) -> PromptVersion | str | list[dict[str, Any]]:
        """
        Format the template with the given keyword arguments.
        By default, it raises an error if there are missing variables. To format
        the prompt partially, set `allow_partial=True`.

        Example:

        .. code-block:: python

            # Text prompt formatting
            prompt = PromptVersion("my-prompt", 1, "Hello, {{title}} {{name}}!")
            formatted = prompt.format(title="Ms", name="Alice")
            print(formatted)
            # Output: "Hello, Ms Alice!"

            # Chat prompt formatting
            chat_prompt = PromptVersion(
                "assistant",
                1,
                [
                    {"role": "system", "content": "You are a {{style}} assistant."},
                    {"role": "user", "content": "{{question}}"},
                ],
            )
            formatted = chat_prompt.format(style="friendly", question="How are you?")
            print(formatted)
            # Output: [{"role": "system", "content": "You are a friendly assistant."},
            #          {"role": "user", "content": "How are you?"}]

            # Partial formatting
            formatted = prompt.format(title="Ms", allow_partial=True)
            print(formatted)
            # Output: PromptVersion(name=my-prompt, version=1, template="Hello, Ms {{name}}!")


        Args:
            allow_partial: If True, allow partial formatting of the prompt text.
                If False, raise an error if there are missing variables.
            kwargs: Keyword arguments to replace the variables in the template.
        """
        input_keys = set(kwargs.keys())
        template_str = self.template if self.is_text_prompt else json.dumps(self.template)
        for key, value in kwargs.items():
            template_str = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), template_str)
        template = template_str if self.is_text_prompt else json.loads(template_str)

        if missing_keys := self.variables - input_keys:
            if not allow_partial:
                raise MlflowException.invalid_parameter_value(
                    f"Missing variables: {missing_keys}. To partially format the prompt, "
                    "set `allow_partial=True`."
                )
            else:
                return PromptVersion(
                    name=self.name,
                    version=int(self.version),
                    template=template,
                    response_format=self.response_format,
                    commit_message=self.commit_message,
                    creation_timestamp=self.creation_timestamp,
                    tags=self.tags,
                    aliases=self.aliases,
                    last_updated_timestamp=self.last_updated_timestamp,
                    user_id=self.user_id,
                )
        return template
