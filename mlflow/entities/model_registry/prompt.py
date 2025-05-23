from __future__ import annotations

import re
from typing import Optional, Union, Dict, List

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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


# Prompt is implemented as a special type of ModelVersion. MLflow stores both prompts
# and model versions in the model registry as ModelVersion DB records, but distinguishes
# them using the special tag "mlflow.prompt.is_prompt".
class Prompt:
    """
    MLflow entity for Unity Catalog Prompt.
    """
    
    def __init__(
        self,
        name: str,
        creation_timestamp: Optional[int] = None,
        last_updated_timestamp: Optional[int] = None,
        description: Optional[str] = None,
        experiment_id: Optional[str] = None,
        aliases: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        latest_version: Optional[PromptVersion] = None,
    ):
        """
        Initialize a Prompt.
        
        Args:
            name: Full three tier UC name of the prompt
            creation_timestamp: Timestamp recorded when this prompt was created
            last_updated_timestamp: Timestamp recorded when metadata was last updated
            description: Description of this prompt
            experiment_id: ExperimentID associated with this prompt
            aliases: Dictionary of alias name to version number
            tags: Dictionary of prompt tags
            latest_version: Latest version of this prompt
        """
        self._name = name
        self._creation_timestamp = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._experiment_id = experiment_id
        self._aliases = aliases or {}
        self._tags = tags or {}
        self._latest_version = latest_version

    @property
    def name(self) -> str:
        """Get the prompt name."""
        return self._name

    @property
    def creation_timestamp(self) -> Optional[int]:
        """Get the creation timestamp."""
        return self._creation_timestamp

    @property
    def last_updated_timestamp(self) -> Optional[int]:
        """Get the last updated timestamp."""
        return self._last_updated_timestamp

    @property
    def description(self) -> Optional[str]:
        """Get the description."""
        return self._description

    @property
    def experiment_id(self) -> Optional[str]:
        """Get the experiment ID."""
        return self._experiment_id

    @property
    def aliases(self) -> Dict[str, str]:
        """Get the aliases dictionary."""
        return self._aliases

    @property
    def tags(self) -> Dict[str, str]:
        """Get the tags dictionary."""
        return self._tags

    @property
    def latest_version(self) -> Optional[PromptVersion]:
        """Get the latest version."""
        return self._latest_version

    @classmethod
    def from_proto(cls, proto, latest_version=None):
        """
        Create a Prompt from protocol buffer.
        
        Args:
            proto: Protocol buffer message
            latest_version: Optional latest version to include
            
        Returns:
            Prompt object
        """
        aliases = {}
        if proto.aliases:
            aliases = {a.alias: a.version for a in proto.aliases}
            
        tags = {}
        if proto.tags:
            tags = {t.key: t.value for t in proto.tags}
            
        creation_ts = None
        if proto.creation_timestamp:
            creation_ts = int(proto.creation_timestamp.seconds * 1000)
            
        last_updated_ts = None
        if proto.last_updated_timestamp:
            last_updated_ts = int(proto.last_updated_timestamp.seconds * 1000)
            
        return cls(
            name=proto.name,
            creation_timestamp=creation_ts,
            last_updated_timestamp=last_updated_ts,
            description=proto.description,
            experiment_id=proto.experiment_id,
            aliases=aliases,
            tags=tags,
            latest_version=latest_version,
        )

    def to_proto(self):
        """
        Convert to protocol buffer message.
        
        Returns:
            Proto message
        """
        from mlflow.protos import uc_prompt_pb2

        proto = uc_prompt_pb2.Prompt()
        proto.name = self.name
        
        if self.description:
            proto.description = self.description
            
        if self.experiment_id:
            proto.experiment_id = self.experiment_id
            
        for alias, version in self.aliases.items():
            proto_alias = proto.aliases.add()
            proto_alias.alias = alias
            proto_alias.version = version
            
        for key, value in self.tags.items():
            proto_tag = proto.tags.add()
            proto_tag.key = key
            proto_tag.value = value
            
        return proto

    def __repr__(self):
        return f"<Prompt: name={self.name}>"

    def __repr__(self) -> str:
        text = (
            self.template[:PROMPT_TEXT_DISPLAY_LIMIT] + "..."
            if len(self.template) > PROMPT_TEXT_DISPLAY_LIMIT
            else self.template
        )
        return f"Prompt(name={self.name}, version={self.version}, template={text})"

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
        return set(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(self.template))

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
                return Prompt(
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
    ) -> Prompt:
        """
        Create a Prompt object from a ModelVersion object.

        Args:
            model_version: The ModelVersion object to convert to a Prompt.
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
