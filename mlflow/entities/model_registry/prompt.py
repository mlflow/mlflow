from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException

# A special tag in RegisteredModel to indicate that it is a prompt
IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"
# A special tag in ModelVersion to store the prompt text
PROMPT_TEXT_TAG_KEY = "mlflow.prompt.text"

_PROMPT_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")

# Alias type
PromptVersionTag = ModelVersionTag


@dataclass
class Prompt(ModelVersion):
    """
    An entity representing a prompt (template) for GenAI applications.

    Args:
        name: The name of the prompt.
        version: The version number of the prompt.
        template: The template text of the prompt. It can contain variables enclosed in
            single curly braces, e.g. {variable}, which will be replaced with actual values
            by the `format` method.
        description: Text description of the prompt. Optional.
        creation_timestamp: Timestamp of the prompt creation. Optional.
        tags: A dictionary of tags associated with the prompt. Optional.

    :meta private:
    Prompt is implemented as a special type of ModelVersion. MLflow stores both prompts and
    model versions in the model registry as ModelVersion DB records, but distinguishes them
    using the special tag "mlflow.prompt.is_prompt".
    """

    def __init__(
        self,
        name: str,
        version: int,
        template: str,
        description: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        # Store template text as a tag
        tags = tags or {}

        if PROMPT_TEXT_TAG_KEY not in tags:
            tags[PROMPT_TEXT_TAG_KEY] = template

        super().__init__(
            name=name,
            version=version,
            creation_timestamp=creation_timestamp,
            description=description,
            tags=[ModelVersionTag(key=key, value=value) for key, value in tags.items()],
        )

    def __repr__(self) -> str:
        text = self.template[:30] + "..." if len(self.template) > 30 else self.template
        return f"Prompt(name={self.name}, version={self.version}, template={text})"

    @property
    def template(self) -> str:
        """
        Return the template text of the prompt.
        """
        return self._tags[PROMPT_TEXT_TAG_KEY]

    @property
    def variables(self) -> set[str]:
        """
        Return a list of variables in the template text.
        The value must be enclosed in curly braces, e.g. {variable}.
        """
        if hasattr(self, "_variables"):
            return self._variables

        variables = _PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(self.template)
        self._variables = set(variables)
        return self._variables

    @property
    def tags(self) -> dict[str, str]:
        """Return the tags of the prompt as a dictionary."""
        # Remove the prompt text tag as it should not be user-facing
        return {key: value for key, value in self._tags.items() if key != PROMPT_TEXT_TAG_KEY}

    @tags.setter
    def tags(self, tags: dict[str, str]):
        """Set the tags of the prompt."""
        self._tags = {
            **tags,
            PROMPT_TEXT_TAG_KEY: self.template,
        }

    def format(self, allow_partial: bool = False, **kwargs) -> Union[Prompt, str]:
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
            # Output: Prompt(name=my-prompt, version=1, template="Hello, Ms {{name}}!")


        Args:
            allow_partial: If True, allow partial formatting of the prompt text.
                If False, raise an error if there are missing variables.
            kwargs: Keyword arguments to replace the variables in the template.
        """
        input_keys = set(kwargs.keys())
        missing_keys = self.variables - input_keys

        if missing_keys and not allow_partial:
            raise MlflowException.invalid_parameter_value(
                f"Missing variables: {missing_keys}. To partially format the prompt, "
                "set `allow_partial=True`."
            )

        return self.template.format(**kwargs)

    @classmethod
    def from_model_version(cls, model_version: ModelVersion) -> Prompt:
        """
        Create a Prompt object from a ModelVersion object.
        """
        if PROMPT_TEXT_TAG_KEY not in model_version.tags:
            raise ValueError("ModelVersion object does not contain prompt text.")

        return cls(
            name=model_version.name,
            version=model_version.version,
            template=model_version.tags[PROMPT_TEXT_TAG_KEY],
            description=model_version.description,
            creation_timestamp=model_version.creation_timestamp,
            tags=model_version.tags,
        )
