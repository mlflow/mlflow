from __future__ import annotations

import re
from typing import Optional, Union

from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_TEMPLATE_VARIABLE_PATTERN,
    PROMPT_TEXT_DISPLAY_LIMIT,
    PROMPT_TEXT_TAG_KEY,
)

# Alias type
PromptVersionTag = ModelVersionTag


def _is_reserved_tag(key: str) -> bool:
    return key in {IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY}


class PromptVersion(_ModelRegistryEntity):
    """
    An entity representing a specific version of a prompt with its template content.

    Args:
        name: The name of the prompt.
        version: The version number of the prompt.
        template: The template text of the prompt. It can contain variables enclosed in
            double curly braces, e.g. {{variable}}, which will be replaced with actual values
            by the `format` method. MLflow use the same variable naming rules same as Jinja2
            https://jinja.palletsprojects.com/en/stable/api/#notes-on-identifiers
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
        template: str,
        commit_message: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        tags: Optional[dict[str, str]] = None,
        aliases: Optional[list[str]] = None,
        last_updated_timestamp: Optional[int] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__()

        # Core PromptVersion attributes
        self._name: str = name
        self._version: str = str(version)  # Store as string internally
        self._creation_time: int = creation_timestamp or 0

        # Store template text as a tag
        tags = tags or {}
        tags[PROMPT_TEXT_TAG_KEY] = template
        tags[IS_PROMPT_TAG_KEY] = "true"

        # Store the tags dict
        self._tags: dict[str, str] = tags

        self._variables = set(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(template))
        self._last_updated_timestamp: Optional[int] = last_updated_timestamp
        self._description: Optional[str] = commit_message
        self._user_id: Optional[str] = user_id
        self._aliases: list[str] = aliases or []

    def __repr__(self) -> str:
        text = (
            self.template[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
            if len(self.template) > PROMPT_TEXT_DISPLAY_LIMIT
            else self.template
        )
        return f"PromptVersion(name={self.name}, version={self.version}, template={text})"

    # Core PromptVersion properties
    @property
    def template(self) -> str:
        """
        Return the template text of the prompt.
        """
        return self._tags[PROMPT_TEXT_TAG_KEY]

    def to_single_brace_format(self) -> str:
        """
        Convert the template text to single brace format. This is useful for integrating with other
        systems that use single curly braces for variable replacement, such as LangChain's prompt
        template. Default is False.
        """
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
                    commit_message=self.commit_message,
                    creation_timestamp=self.creation_timestamp,
                    tags=self.tags,
                    aliases=self.aliases,
                    last_updated_timestamp=self.last_updated_timestamp,
                    user_id=self.user_id,
                )
        return template
