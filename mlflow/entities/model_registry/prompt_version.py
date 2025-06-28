from __future__ import annotations

import json
import re
from typing import Optional, Union

from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    CONFIG_TAG_KEY,
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
        CONFIG_TAG_KEY,
    }


class PromptVersion(_ModelRegistryEntity):
    """
    An entity representing a specific version of a prompt with its template content.

    Args:
        name: The name of the prompt.
        version: The version number of the prompt.
        template: The template text of the prompt. For text prompts, it can contain variables
        enclosed in double curly braces, e.g. {{variable}}. For chat prompts, it should be a list
        of message dictionaries with 'role' and 'content' fields.
        prompt_type: The type of prompt ("text" or "chat"). Defaults to "text".
        response_format: Optional Pydantic class defining the response structure. The JSON schema
        will be stored and can be used for API integration.
        config: Optional model configuration dictionary.
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
        template: Union[str, list[dict[str, str]]],
        prompt_type: str = "text",
        response_format: Optional[Union[dict, type]] = None,
        config: Optional[dict] = None,
        commit_message: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        tags: Optional[dict[str, str]] = None,
        aliases: Optional[list[str]] = None,
        last_updated_timestamp: Optional[int] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__()

        # Validate prompt_type
        if prompt_type is None:
            prompt_type = PROMPT_TYPE_TEXT
        elif prompt_type not in [PROMPT_TYPE_TEXT, PROMPT_TYPE_CHAT]:
            raise ValueError("prompt_type must be 'text' or 'chat'")

        # Validate chat template structure
        if prompt_type == PROMPT_TYPE_CHAT:
            if not isinstance(template, list):
                raise ValueError("Chat template must be a list of message dictionaries")

            if len(template) == 0:
                raise ValueError("Chat template must be a list of message dictionaries")

            # Validate each message has required keys
            for message in template:
                if not isinstance(message, dict):
                    raise ValueError("Each message must be a dictionary")
                if "role" not in message or "content" not in message:
                    raise ValueError("Each message must have 'role' and 'content' keys")
                if message["role"] not in ["system", "user", "assistant"]:
                    raise ValueError("Role must be one of: system, user, assistant")

        # Core PromptVersion attributes
        self._name: str = name
        self._version: str = str(version)  # Store as string internally
        self._creation_time: int = creation_timestamp or 0

        # Store template text as a tag
        tags = tags or {}
        if prompt_type == PROMPT_TYPE_CHAT:
            # For chat prompts, store the template as JSON
            tags[PROMPT_TEXT_TAG_KEY] = json.dumps(template)
        else:
            # For text prompts, store as string
            tags[PROMPT_TEXT_TAG_KEY] = template
        tags[IS_PROMPT_TAG_KEY] = "true"

        # Store new fields as tags
        if prompt_type != PROMPT_TYPE_TEXT:
            tags[PROMPT_TYPE_TAG_KEY] = prompt_type
        if response_format:
            # Store the response format as a JSON schema
            tags[RESPONSE_FORMAT_TAG_KEY] = json.dumps(
                self._convert_response_format_to_dict(response_format)
            )
        if config:
            tags[CONFIG_TAG_KEY] = json.dumps(config)

        # Store the tags dict
        self._tags: dict[str, str] = tags

        # Handle variables for text prompts
        if prompt_type == PROMPT_TYPE_TEXT:
            self._variables = set(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(template))
        else:
            # For chat prompts, extract variables from all message contents
            self._variables = set()
            for message in template:
                if isinstance(message, dict) and "content" in message:
                    self._variables.update(
                        PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(message["content"])
                    )

        self._last_updated_timestamp: Optional[int] = last_updated_timestamp
        self._description: Optional[str] = commit_message
        self._user_id: Optional[str] = user_id
        self._aliases: list[str] = aliases or []

    def _convert_response_format_to_dict(self, response_format: Union[dict, type]) -> dict:
        """Convert Pydantic class to JSON schema format."""
        if isinstance(response_format, dict):
            # Already a dict, return as is (for backward compatibility)
            return response_format
        elif isinstance(response_format, type):
            try:
                from pydantic import BaseModel

                if issubclass(response_format, BaseModel):
                    # Return the JSON schema for the Pydantic model
                    return response_format.model_json_schema()
                else:
                    raise ValueError("response_format must be a Pydantic class or dict")
            except ImportError:
                # Pydantic not available, return basic schema
                return {"type": "object", "properties": {}}
        else:
            raise ValueError("response_format must be a Pydantic class or dict")

    def __repr__(self) -> str:
        if self.prompt_type == PROMPT_TYPE_CHAT:
            text = f"Chat prompt with {len(self.template)} messages"
        else:
            text = (
                self.template[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
                if len(self.template) > PROMPT_TEXT_DISPLAY_LIMIT
                else self.template
            )
        return f"PromptVersion(name={self.name}, version={self.version}, template={text})"

    # Core PromptVersion properties
    @property
    def template(self) -> Union[str, list[dict[str, str]]]:
        """
        Return the template of the prompt.
        For text prompts, returns a string.
        For chat prompts, returns a list of message dictionaries.
        """
        template_str = self._tags[PROMPT_TEXT_TAG_KEY]
        if self.prompt_type == PROMPT_TYPE_CHAT:
            return json.loads(template_str)
        return template_str

    @property
    def prompt_type(self) -> str:
        """Return the type of prompt (text or chat)."""
        return self._tags.get(PROMPT_TYPE_TAG_KEY, PROMPT_TYPE_TEXT)

    @property
    def response_format(self) -> Optional[dict]:
        """Return the response format JSON schema if defined."""
        format_str = self._tags.get(RESPONSE_FORMAT_TAG_KEY)
        return json.loads(format_str) if format_str else None

    @property
    def config(self) -> Optional[dict]:
        """Return the model configuration if defined."""
        config_str = self._tags.get(CONFIG_TAG_KEY)
        return json.loads(config_str) if config_str else None

    def format_chat(self, **kwargs) -> list[dict[str, str]]:
        """Format chat-style prompts with variable substitution."""
        if self.prompt_type != PROMPT_TYPE_CHAT:
            raise MlflowException("format_chat() can only be used with chat prompts")

        # Handle chat template formatting
        formatted_messages = []
        for message in self.template:
            content = message["content"]
            for key, value in kwargs.items():
                content = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), content)
            formatted_messages.append({"role": message["role"], "content": content})
        return formatted_messages

    def to_single_brace_format(self) -> str:
        """
        Convert the template text to single brace format. This is useful for integrating with other
        systems that use single curly braces for variable replacement, such as LangChain's prompt
        template. Default is False.
        """
        if self.prompt_type != PROMPT_TYPE_TEXT:
            raise MlflowException("to_single_brace_format() can only be used with text prompts")

        t = self.template
        for var in self.variables:
            t = re.sub(r"\{\{\s*" + var + r"\s*\}\}", "{" + var + "}", t)
        return t

    @property
    def variables(self) -> set[str]:
        """
        Return a list of variables in the template text.
        The value must be enclosed in double curly braces, e.g. {{variable}}.
        """
        return self._variables

    @property
    def commit_message(self) -> Optional[str]:
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
    def last_updated_timestamp(self) -> Optional[int]:
        """Integer. Timestamp of last update for this prompt version (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def description(self) -> Optional[str]:
        """String. Description"""
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description

    @property
    def user_id(self) -> Optional[str]:
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

    def format(self, allow_partial: bool = False, **kwargs) -> Union[PromptVersion, str]:
        """
        Format the template text with the given keyword arguments.
        By default, it raises an error if there are missing variables. To format
        the prompt text partially, set `allow_partial=True`.

        For chat prompts, use format_chat() method instead.

        Example:

        .. code-block:: python

            prompt = Prompt("my-prompt", 1, "Hello, {{title}} {{name}}!")
            formatted = prompt.format(title="Ms", name="Alice")
            print(formatted)
            # Output: "Hello, Ms Alice!"

            # Partial formatting
            formatted = prompt.format(title="Ms", allow_partial=True)
            print(formatted)
            # Output: PromptVersion(name=my-prompt, version=1, template="Hello, Ms {{name}}!")


        Args:
            allow_partial: If True, allow partial formatting of the prompt text.
                If False, raise an error if there are missing variables.
            kwargs: Keyword arguments to replace the variables in the template.
        """
        if self.prompt_type == PROMPT_TYPE_CHAT:
            raise MlflowException(
                "format() cannot be used with chat prompts. Use format_chat() instead."
            )

        input_keys = set(kwargs.keys())

        template = self.template
        for key, value in kwargs.items():
            template = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), template)

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
                    prompt_type=self.prompt_type,
                    response_format=self.response_format,
                    config=self.config,
                    commit_message=self.commit_message,
                    creation_timestamp=self.creation_timestamp,
                    tags=self.tags,
                    aliases=self.aliases,
                    last_updated_timestamp=self.last_updated_timestamp,
                    user_id=self.user_id,
                )
        return template
