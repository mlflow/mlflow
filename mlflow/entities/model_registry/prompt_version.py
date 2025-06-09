from __future__ import annotations

import re
from typing import Optional, Union

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY,
    PROMPT_TEMPLATE_VARIABLE_PATTERN,
    PROMPT_TEXT_DISPLAY_LIMIT,
    PROMPT_TEXT_TAG_KEY,
)

# Alias type
PromptVersionTag = ModelVersionTag


def _is_reserved_tag(key: str) -> bool:
    return key in {IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY}


class PromptVersion(ModelVersion):
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
        version_metadata: A dictionary of metadata associated with the **prompt version**.
            This is useful for storing version-specific information, such as the author of
            the changes. Optional.
        prompt_tags: A dictionary of tags associated with the entire prompt. This is different
            from the `version_metadata` as it is not tied to a specific version of the prompt.
    """

    def __init__(
        self,
        name: str,
        version: int,
        template: str,
        commit_message: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        version_metadata: Optional[dict[str, str]] = None,
        prompt_tags: Optional[dict[str, str]] = None,
        aliases: Optional[list[str]] = None,
    ):
        # Store template text as a tag
        version_metadata = version_metadata or {}
        version_metadata[PROMPT_TEXT_TAG_KEY] = template
        version_metadata[IS_PROMPT_TAG_KEY] = "true"

        super().__init__(
            name=name,
            version=version,
            creation_timestamp=creation_timestamp,
            description=commit_message,
            # "version_metadata" is represented as ModelVersion tags.
            tags=[ModelVersionTag(key=key, value=value) for key, value in version_metadata.items()],
            aliases=aliases,
        )

        self._variables = set(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(self.template))

        # Store the prompt-level tags (from RegisteredModel).
        self._prompt_tags = prompt_tags or {}

    def __repr__(self) -> str:
        text = (
            self.template[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
            if len(self.template) > PROMPT_TEXT_DISPLAY_LIMIT
            else self.template
        )
        return f"PromptVersion(name={self.name}, version={self.version}, template={text})"

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
        return self.description  # inherited from ModelVersion

    @property
    def version_metadata(self) -> dict[str, str]:
        """Return the tags of the prompt as a dictionary."""
        # Remove the prompt text tag as it should not be user-facing
        return {key: value for key, value in self._tags.items() if not _is_reserved_tag(key)}

    @property
    def tags(self) -> dict[str, str]:
        """
        Return the prompt-level tags (from RegisteredModel).
        """
        return {key: value for key, value in self._prompt_tags.items() if not _is_reserved_tag(key)}

    @property
    def run_ids(self) -> list[str]:
        """Get the run IDs associated with the prompt."""
        run_tag = self._tags.get(PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY)
        if not run_tag:
            return []
        return run_tag.split(",")

    @property
    def uri(self) -> str:
        """Return the URI of the prompt."""
        return f"prompts:/{self.name}/{self.version}"

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
                    version=self.version,
                    template=template,
                    commit_message=self.commit_message,
                    creation_timestamp=self.creation_timestamp,
                    prompt_tags=self._prompt_tags,
                    version_metadata=self.version_metadata,
                    aliases=self.aliases,
                )
        return template

    @classmethod
    def from_model_version(
        cls, model_version: ModelVersion, prompt_tags: Optional[dict[str, str]] = None
    ) -> PromptVersion:
        """
        Create a PromptVersion object from a ModelVersion object.

        Args:
            model_version: The ModelVersion object to convert to a PromptVersion.
            prompt_tags: The prompt-level tags (from RegisteredModel). Optional.
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

        return cls(
            name=model_version.name,
            version=model_version.version,
            template=model_version.tags[PROMPT_TEXT_TAG_KEY],
            commit_message=model_version.description,
            creation_timestamp=model_version.creation_timestamp,
            version_metadata=model_version.tags,
            prompt_tags=prompt_tags,
            aliases=model_version.aliases,
        )

    # Override model-specific properties that don't make sense for prompts
    @property
    def run_id(self):
        """MLflow run that generated the model - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'run_id'. Prompts are not tied to specific MLflow runs. "
            "Use 'run_ids' property for associated runs."
        )

    @property
    def run_link(self):
        """Link to the MLflow run - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'run_link'. Prompts are not tied to specific MLflow runs."
        )

    @property
    def model_id(self):
        """ID linking to logged model - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'model_id'. Prompts are separate from models."
        )

    @property
    def params(self):
        """Model parameters - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'params'. Prompts do not have model parameters."
        )

    @property
    def metrics(self):
        """Model metrics - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'metrics'. Prompts do not have model metrics."
        )

    @property
    def deployment_job_state(self):
        """Deployment job information - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'deployment_job_state'. Prompts are not deployed like models."
        )

    @property
    def last_updated_timestamp(self):
        """Timestamp of last update - inherited but not relevant for prompts."""
        raise AttributeError(
            "PromptVersion does not have 'last_updated_timestamp'. Use 'creation_timestamp' instead."
        )

    @property
    def current_stage(self):
        """Model staging information - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'current_stage'. Prompts do not use staging concepts like models."
        )

    @property
    def source(self):
        """Model artifact source path - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'source'. Prompts do not have artifact source paths."
        )

    @property
    def status(self):
        """Model deployment status - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'status'. Prompts do not have deployment status."
        )

    @property
    def status_message(self):
        """Model deployment status message - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'status_message'. Prompts do not have deployment status."
        )

    @property
    def user_id(self):
        """User who created the model - not applicable to prompts."""
        raise AttributeError(
            "PromptVersion does not have 'user_id'. User information is not tracked for prompts."
        )
