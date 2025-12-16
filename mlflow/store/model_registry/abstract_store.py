import json
import logging
import re
import threading
from abc import ABCMeta, abstractmethod
from time import sleep, time
from typing import Any

from pydantic import BaseModel

from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import (
    PromptModelConfig,
    PromptVersion,
)
from mlflow.entities.webhook import Webhook, WebhookEvent, WebhookStatus, WebhookTestResult
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_EXPERIMENT_IDS_TAG_KEY,
    PROMPT_MODEL_CONFIG_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.prompt.registry_utils import has_prompt_tag, model_version_to_prompt_version
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.utils.prompt import update_linked_prompts_tag
from mlflow.utils.annotations import developer_stable
from mlflow.utils.logging_utils import eprint

_logger = logging.getLogger(__name__)

AWAIT_MODEL_VERSION_CREATE_SLEEP_INTERVAL_SECONDS = 3


@developer_stable
class AbstractStore:
    """
    Abstract class that defines API interfaces for storing Model Registry metadata.
    """

    __metaclass__ = ABCMeta

    def __init__(self, store_uri=None, tracking_uri=None):
        """
        Empty constructor. This is deliberately not marked as abstract, else every derived class
        would be forced to create one.

        Args:
            store_uri: The model registry store URI.
            tracking_uri: URI of the current MLflow tracking server, used to perform operations
                like fetching source run metadata or downloading source run artifacts
                to support subsequently uploading them to the model registry storage
                location.
        """
        # Create a thread lock to ensure thread safety when linking prompts to other entities,
        # since the default linking implementation reads and appends entity tags, which
        # is prone to concurrent modification issues
        self._prompt_link_lock = threading.RLock()

    def __getstate__(self):
        """Support for pickle serialization by excluding the non-picklable RLock."""
        state = self.__dict__.copy()
        # Remove the RLock as it cannot be pickled
        del state["_prompt_link_lock"]
        return state

    def __setstate__(self, state):
        """Support for pickle deserialization by recreating the RLock."""
        self.__dict__.update(state)
        # Recreate the RLock
        self._prompt_link_lock = threading.RLock()

    # CRUD API for RegisteredModel objects

    @abstractmethod
    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.

        """

    @abstractmethod
    def update_registered_model(self, name, description, deployment_job_id=None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """

    @abstractmethod
    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """

    @abstractmethod
    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """

    @abstractmethod
    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.
        """

    @abstractmethod
    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """

    @abstractmethod
    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                each stage.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """

    @abstractmethod
    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """

    @abstractmethod
    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """

    # CRUD API for ModelVersion objects

    @abstractmethod
    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
        model_id: str | None = None,
    ):
        """
        Create a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            local_model_path: Local path to the MLflow model, if it's already accessible
                on the local filesystem. Can be used by AbstractStores that
                upload model version files to the model registry to avoid
                a redundant download from the source location when logging
                and registering a model via a single
                mlflow.<flavor>.log_model(..., registered_model_name) call
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                registered model version, if applicable.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.

        """

    @abstractmethod
    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """

    @abstractmethod
    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will
                be raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """

    @abstractmethod
    def delete_model_version(self, name, version):
        """
        Delete model model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """

    @abstractmethod
    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """

    @abstractmethod
    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single URI location that allows reads for downloading.
        """

    @abstractmethod
    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or
                ``run_id = '...'``.
            max_results: Maximum number of model versions desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_model_versions`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects that satisfy the search expressions. The pagination token for the next
            page can be obtained via the ``token`` attribute of the object.

        """

    @abstractmethod
    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.

        Returns:
            None
        """

    @abstractmethod
    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """

    @abstractmethod
    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """

    @abstractmethod
    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """

    @abstractmethod
    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """

    def copy_model_version(self, src_mv, dst_name):
        """
        Copy a model version from one registered model to another as a new model version.

        Args:
            src_mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                the source model version.
            dst_name: The name of the registered model to copy the model version to. If a
                registered model with this name does not exist, it will be created.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
            the cloned model version.
        """
        try:
            create_model_response = self.create_registered_model(dst_name)
            eprint(f"Successfully registered model '{create_model_response.name}'.")
        except MlflowException as e:
            if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise
            eprint(
                f"Registered model '{dst_name}' already exists."
                f" Creating a new version of this model..."
            )

        try:
            mv_copy = self.create_model_version(
                name=dst_name,
                source=f"models:/{src_mv.name}/{src_mv.version}",
                run_id=src_mv.run_id,
                tags=[ModelVersionTag(k, v) for k, v in src_mv.tags.items()],
                run_link=src_mv.run_link,
                description=src_mv.description,
                model_id=src_mv.model_id,
            )
            eprint(
                f"Copied version '{src_mv.version}' of model '{src_mv.name}'"
                f" to version '{mv_copy.version}' of model '{mv_copy.name}'."
            )
        except MlflowException as e:
            raise MlflowException(
                f"Failed to create model version copy. The current model registry backend "
                f"may not yet support model version URI sources.\nError: {e}"
            ) from e

        return mv_copy

    def _await_model_version_creation(self, mv, await_creation_for):
        """
        Await for model version to become ready after creation.

        Args:
            mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object.
            await_creation_for: Number of seconds to wait for the model version to finish being
                created and is in ``READY`` status.
        """
        self._await_model_version_creation_impl(mv, await_creation_for)

    def _await_model_version_creation_impl(self, mv, await_creation_for, hint=""):
        entity_type = "Prompt" if has_prompt_tag(mv.tags) else "Model"
        _logger.info(
            f"Waiting up to {await_creation_for} seconds for {entity_type.lower()} version to "
            f"finish creation. {entity_type} name: {mv.name}, version {mv.version}",
        )
        max_time = time() + await_creation_for
        pending_status = ModelVersionStatus.to_string(ModelVersionStatus.PENDING_REGISTRATION)
        while mv.status == pending_status:
            if time() > max_time:
                raise MlflowException(
                    f"Exceeded max wait time for model name: {mv.name} version: {mv.version} "
                    f"to become READY. Status: {mv.status} Wait Time: {await_creation_for}"
                    f".{hint}"
                )
            mv = self.get_model_version(mv.name, mv.version)
            if mv.status != pending_status:
                break
            sleep(AWAIT_MODEL_VERSION_CREATE_SLEEP_INTERVAL_SECONDS)
        if mv.status != ModelVersionStatus.to_string(ModelVersionStatus.READY):
            raise MlflowException(
                f"{entity_type} version creation failed for {entity_type.lower()} name: {mv.name} "
                f"version: {mv.version} with status: {mv.status} and message: {mv.status_message}"
            )

    # Prompt-related methods with concrete implementations for OSS stores

    def create_prompt(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Prompt:
        """
        Create a new prompt in the registry.

        Default implementation: creates a RegisteredModel with special prompt tags.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            description: Optional description of the prompt.
            tags: Optional dictionary of prompt tags.

        Returns:
            A Prompt object representing the created prompt.
        """
        # Default implementation: use RegisteredModel with special tags
        prompt_tags = [RegisteredModelTag(key=IS_PROMPT_TAG_KEY, value="true")]
        if tags:
            prompt_tags.extend([RegisteredModelTag(key=k, value=v) for k, v in tags.items()])

        # Create registered model for the prompt
        rm = self.create_registered_model(name, tags=prompt_tags, description=description)

        # Return as Prompt
        return Prompt(
            name=rm.name,
            description=rm.description,
            creation_timestamp=rm.creation_timestamp,
            tags=tags or {},
        )

    @staticmethod
    def _parse_experiment_id_filter(filter_string: str | None) -> str | None:
        """
        Parse and transform experiment_id filter to tag-based filter.

        This helper extracts the special 'experiment_id = "xxx"' syntax from the filter
        string, converts it to the appropriate tag filter clause, and combines it with
        any remaining filters.

        Args:
            filter_string: Original filter string that may contain experiment_id clause

        Returns:
            Transformed filter string with experiment_id converted to tag filter, or None
        """
        if not filter_string:
            return None

        # Match experiment_id = 'xxx' or experiment_id = "xxx"
        exp_id_pattern = r"experiment_id\s*=\s*['\"]([^'\"]+)['\"]"
        match = re.search(exp_id_pattern, filter_string)

        if not match:
            return filter_string

        experiment_id = match.group(1)

        # Remove the experiment_id clause from the filter string
        remaining_filter = re.sub(exp_id_pattern, "", filter_string).strip()

        # Clean up any leading/trailing AND operators
        remaining_filter = re.sub(r"^\s*AND\s+", "", remaining_filter)
        remaining_filter = re.sub(r"\s+AND\s*$", "", remaining_filter)
        remaining_filter = re.sub(r"\s+AND\s+AND\s+", " AND ", remaining_filter)

        # Build the tag filter clause
        if not experiment_id.isdigit():
            raise MlflowException(
                f"Invalid experiment_id: {experiment_id}. Must be a numeric value.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        # Use LIKE to match the experiment ID anywhere in the comma-separated list
        experiment_filter = f"tags.{PROMPT_EXPERIMENT_IDS_TAG_KEY} LIKE '%,{experiment_id},%'"

        if remaining_filter:
            return f"{experiment_filter} AND {remaining_filter}"
        return experiment_filter

    def search_prompts(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in the registry.

        Default implementation: searches RegisteredModels with prompt tags.
        Other store implementations may override this method.

        Args:
            filter_string: Filter query string, defaults to searching for all prompts.
            max_results: Maximum number of prompts desired.
            order_by: List of order-by clauses.
            page_token: Pagination token for requesting subsequent pages.

        Returns:
            A PagedList of Prompt objects.
        """
        if max_results is None:
            max_results = 100

        # Build filter to only include prompts (use backticks for tag key with dots)
        prompt_filter = f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'"

        if filter_string:
            filter_string = self._parse_experiment_id_filter(filter_string)
            prompt_filter = f"{prompt_filter} AND {filter_string}"

        # Search registered models with prompt filter
        registered_models = self.search_registered_models(
            filter_string=prompt_filter,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

        # Convert RegisteredModel objects to Prompt objects
        prompts = []
        for rm in registered_models:
            # Extract tags as dict
            if isinstance(rm.tags, dict):
                tags = rm.tags.copy()
            else:
                tags = {tag.key: tag.value for tag in rm.tags} if rm.tags else {}

            # Remove the internal prompt tag from user-visible tags
            tags.pop(IS_PROMPT_TAG_KEY, None)

            # Create Prompt object
            prompt_info = Prompt(
                name=rm.name,
                description=rm.description,
                creation_timestamp=rm.creation_timestamp,
                tags=tags,
            )
            prompts.append(prompt_info)

        return PagedList(prompts, registered_models.token)

    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from the registry.

        Default implementation: deletes the underlying RegisteredModel.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt to delete.
        """
        # Default implementation: delete the registered model
        return self.delete_registered_model(name)

    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt.

        Default implementation: sets a tag on the underlying RegisteredModel.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            key: Tag key.
            value: Tag value.
        """
        # Default implementation: set tag on registered model
        tag = RegisteredModelTag(key=key, value=value)
        return self.set_registered_model_tag(name, tag)

    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt.

        Default implementation: deletes a tag from the underlying RegisteredModel.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            key: Tag key to delete.
        """
        # Default implementation: delete tag from registered model
        return self.delete_registered_model_tag(name, key)

    def get_prompt(self, name: str) -> Prompt | None:
        """
        Get prompt metadata by name.

        Default implementation: gets RegisteredModel with prompt tags and converts to Prompt.
        Other store implementations may override this method.

        Args:
            name: Registered prompt name.

        Returns:
            A single Prompt object with prompt metadata, or None if not found.
        """
        try:
            rm = self.get_registered_model(name)

            # Check if this is actually a prompt using _tags (internal tags)
            if isinstance(rm._tags, dict):
                internal_tags = rm._tags.copy()
            else:
                internal_tags = {tag.key: tag.value for tag in rm._tags} if rm._tags else {}

            if not internal_tags.get(IS_PROMPT_TAG_KEY) == "true":
                return None

            # Get user-visible tags (without internal prompt tag)
            if isinstance(rm.tags, dict):
                user_tags = rm.tags.copy()
            else:
                user_tags = {tag.key: tag.value for tag in rm.tags} if rm.tags else {}

            return Prompt(
                name=rm.name,
                description=rm.description,
                creation_timestamp=rm.creation_timestamp,
                tags=user_tags,
            )

        except Exception:
            return None

    def create_prompt_version(
        self,
        name: str,
        template: str | list[dict[str, Any]],
        description: str | None = None,
        tags: dict[str, str] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        model_config: PromptModelConfig | dict[str, Any] | None = None,
    ) -> PromptVersion:
        """
        Create a new version of an existing prompt.

        Default implementation: creates a ModelVersion with prompt tags.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            template: The prompt template content. Can be either:
                - A string containing text with variables enclosed in double curly braces,
                  e.g. {{variable}}, which will be replaced with actual values by the `format`
                  method.
                - A list of dictionaries representing chat messages, where each message has
                  'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello {{name}}"}])
            description: Optional description of the prompt version.
            tags: Optional dictionary of version tags.
            response_format: Optional Pydantic class or dictionary defining the expected response
                structure. This can be used to specify the schema for structured outputs from LLM
                calls.
            model_config: Optional PromptModelConfig instance or dictionary containing
                model-specific configuration. Using PromptModelConfig provides validation and type
                safety.

        Returns:
            A PromptVersion object representing the created version.
        """
        # Create version tags including template
        version_tags = [
            ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
        ]
        if isinstance(template, str):
            version_tags.append(ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value=template))
            version_tags.append(ModelVersionTag(key=PROMPT_TYPE_TAG_KEY, value=PROMPT_TYPE_TEXT))
        else:
            version_tags.append(
                ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value=json.dumps(template))
            )
            version_tags.append(ModelVersionTag(key=PROMPT_TYPE_TAG_KEY, value=PROMPT_TYPE_CHAT))
        if response_format:
            version_tags.append(
                ModelVersionTag(
                    key=RESPONSE_FORMAT_TAG_KEY,
                    value=json.dumps(
                        PromptVersion.convert_response_format_to_dict(response_format)
                    ),
                )
            )
        if model_config:
            # Convert PromptModelConfig to dict if needed
            if isinstance(model_config, PromptModelConfig):
                config_dict = model_config.to_dict()
            else:
                # Validate dict by converting through PromptModelConfig
                config_dict = PromptModelConfig.from_dict(model_config).to_dict()

            version_tags.append(
                ModelVersionTag(key=PROMPT_MODEL_CONFIG_TAG_KEY, value=json.dumps(config_dict))
            )

        if tags:
            version_tags.extend([ModelVersionTag(key=k, value=v) for k, v in tags.items()])

        # Create model version
        mv = self.create_model_version(
            name=name,
            source="prompt-template",  # Required field for ModelVersion
            tags=version_tags,
            description=description,
        )

        # Get prompt-level tags from registered model
        rm = self.get_registered_model(name)
        if isinstance(rm.tags, dict):
            prompt_tags = rm.tags.copy()
        else:
            prompt_tags = {tag.key: tag.value for tag in rm.tags}

        return model_version_to_prompt_version(mv, prompt_tags=prompt_tags)

    def get_prompt_version(self, name: str, version: str | int) -> PromptVersion | None:
        """
        Get a specific prompt version.

        Default implementation: gets ModelVersion and converts to PromptVersion.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            version: Version number or alias.

        Returns:
            A PromptVersion object, or None if not found.
        """
        try:
            # First check if this is actually a prompt by checking the registered model
            rm = self.get_registered_model(name)

            # Check if this is actually a prompt using _tags (internal tags)
            if hasattr(rm, "_tags") and isinstance(rm._tags, dict):
                internal_tags = rm._tags.copy()
            elif hasattr(rm, "_tags") and rm._tags:
                internal_tags = {tag.key: tag.value for tag in rm._tags}
            else:
                internal_tags = {}

            if not internal_tags.get(IS_PROMPT_TAG_KEY) == "true":
                raise MlflowException(
                    f"Name `{name}` is registered as a model, not a prompt. "
                    f"Use get_model_version() or load_model() instead.",
                    INVALID_PARAMETER_VALUE,
                )

            # Now get the specific version
            try:
                version_int = int(str(version))
                mv = self.get_model_version(name, version_int)
            except (ValueError, TypeError):
                # Treat as alias
                mv = self.get_model_version_by_alias(name, str(version))

            if not has_prompt_tag(mv.tags):
                return None

            # Get user-visible tags from registered model
            if isinstance(rm.tags, dict):
                prompt_tags = rm.tags.copy()
            else:
                prompt_tags = {tag.key: tag.value for tag in rm.tags}

            return model_version_to_prompt_version(mv, prompt_tags=prompt_tags)

        except MlflowException:
            raise  # Re-raise MlflowExceptions (including our custom one above)
        except Exception:
            return None

    def delete_prompt_version(self, name: str, version: str | int) -> None:
        """
        Delete a specific prompt version.

        Default implementation: deletes the underlying ModelVersion.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            version: Version number to delete.
        """
        # Convert version to int if needed
        try:
            version_int = int(version)
        except (ValueError, TypeError):
            raise MlflowException(f"Invalid version number: {version}")
        return self.delete_model_version(name, version_int)

    def get_prompt_version_by_alias(self, name: str, alias: str) -> PromptVersion | None:
        """
        Get a prompt version by alias.

        Default implementation: uses get_model_version_by_alias and converts to PromptVersion.

        Args:
            name: Name of the prompt.
            alias: Alias name.

        Returns:
            A PromptVersion object, or None if not found.
        """
        return self.get_prompt_version(name, alias)

    def set_prompt_alias(self, name: str, alias: str, version: str | int) -> None:
        """
        Set an alias for a prompt version.

        Default implementation: uses set_registered_model_alias.

        Args:
            name: Name of the prompt.
            alias: Alias to set.
            version: Version to alias.
        """
        self.set_registered_model_alias(name, alias, version)

    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete a prompt alias.

        Default implementation: uses delete_registered_model_alias.

        Args:
            name: Name of the prompt.
            alias: Alias to delete.
        """
        self.delete_registered_model_alias(name, alias)

    def search_prompt_versions(
        self, name: str, max_results: int | None = None, page_token: str | None = None
    ):
        """
        Search prompt versions for a given prompt name.

        This method is only supported in Unity Catalog registries.
        For OSS registries, this functionality is not available.

        Args:
            name: Name of the prompt to search versions for
            max_results: Maximum number of versions to return
            page_token: Token for pagination

        Raises:
            MlflowException: Always, as this is not supported in OSS registries
        """
        raise MlflowException(
            "search_prompt_versions() is not supported in this registry. "
            "This method is only available in Unity Catalog registries.",
            INVALID_PARAMETER_VALUE,
        )

    def link_prompts_to_trace(self, prompt_versions: list[PromptVersion], trace_id: str) -> None:
        """
        Link multiple prompt versions to a trace.

        Default implementation sets a tag on the trace. Stores can override with custom behavior.

        Args:
            prompt_versions: List of PromptVersion objects to link.
            trace_id: Trace ID to link to each prompt version.
        """
        from mlflow.tracing.client import TracingClient
        from mlflow.tracking import _get_store as _get_tracking_store

        client = TracingClient()
        with self._prompt_link_lock:
            trace_info = client.get_trace_info(trace_id)
            if not trace_info:
                raise MlflowException(
                    f"Could not find trace with ID '{trace_id}' to which to link prompts.",
                    error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
                )

            # Try to use the tracking store's EntityAssociation-based linking
            tracking_store = _get_tracking_store()
            try:
                tracking_store.link_prompts_to_trace(trace_id, prompt_versions)
            except NotImplementedError:
                _logger.debug(
                    f"Linking prompts to trace {trace_id} failed. "
                    "Tracking store does not support `link_prompts_to_trace` method."
                )
            finally:
                # Use utility function to update linked prompts tag
                current_tag_value = trace_info.tags.get(TraceTagKey.LINKED_PROMPTS)
                updated_tag_value = update_linked_prompts_tag(current_tag_value, prompt_versions)

                # Only update if the tag value actually changed (avoiding redundant updates)
                if current_tag_value != updated_tag_value:
                    client.set_trace_tag(
                        trace_id,
                        TraceTagKey.LINKED_PROMPTS,
                        updated_tag_value,
                    )

    def set_prompt_version_tag(self, name: str, version: str | int, key: str, value: str) -> None:
        """
        Set a tag on a prompt version.

        Default implementation: uses set_model_version_tag on the underlying ModelVersion.
        Unity Catalog store implementations may override this method.

        Args:
            name: Name of the prompt.
            version: Version number of the prompt.
            key: Tag key.
            value: Tag value.
        """
        # Convert version to int if needed
        try:
            version_int = int(version)
        except (ValueError, TypeError):
            raise MlflowException(f"Invalid version number: {version}")

        # Create a ModelVersionTag and delegate to the underlying model version method
        tag = ModelVersionTag(key=key, value=value)
        return self.set_model_version_tag(name, version_int, tag)

    def delete_prompt_version_tag(self, name: str, version: str | int, key: str) -> None:
        """
        Delete a tag from a prompt version.

        Default implementation: uses delete_model_version_tag on the underlying ModelVersion.
        Unity Catalog store implementations may override this method.

        Args:
            name: Name of the prompt.
            version: Version number of the prompt.
            key: Tag key to delete.
        """
        # Convert version to int if needed
        try:
            version_int = int(version)
        except (ValueError, TypeError):
            raise MlflowException(f"Invalid version number: {version}")

        # Delegate to the underlying model version method
        return self.delete_model_version_tag(name, version_int, key)

    def link_prompt_version_to_model(self, name: str, version: str, model_id: str) -> None:
        """
        Link a prompt version to a model.

        Default implementation sets a tag. Stores can override with custom behavior.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            model_id: ID of the model to link to.
        """
        from mlflow.tracking import _get_store as _get_tracking_store

        prompt_version = self.get_prompt_version(name, version)
        tracking_store = _get_tracking_store()

        with self._prompt_link_lock:
            logged_model = tracking_store.get_logged_model(model_id)
            if not logged_model:
                raise MlflowException(
                    f"Could not find model with ID '{model_id}' to which to link prompt '{name}'.",
                    error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
                )

            current_tag_value = logged_model.tags.get(TraceTagKey.LINKED_PROMPTS)
            updated_tag_value = update_linked_prompts_tag(current_tag_value, [prompt_version])

            if current_tag_value != updated_tag_value:
                tracking_store.set_logged_model_tags(
                    model_id,
                    [
                        LoggedModelTag(
                            key=TraceTagKey.LINKED_PROMPTS,
                            value=updated_tag_value,
                        )
                    ],
                )

    def link_prompt_version_to_run(self, name: str, version: str, run_id: str) -> None:
        """
        Link a prompt version to a run.

        Default implementation sets a tag. Stores can override with custom behavior.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            run_id: ID of the run to link to.
        """
        from mlflow.tracking import _get_store as _get_tracking_store

        prompt_version = self.get_prompt_version(name, version)
        tracking_store = _get_tracking_store()

        with self._prompt_link_lock:
            run = tracking_store.get_run(run_id)
            if not run:
                raise MlflowException(
                    f"Could not find run with ID '{run_id}' to which to link prompt '{name}'.",
                    error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
                )

            current_tag_value = None
            if isinstance(run.data.tags, dict):
                current_tag_value = run.data.tags.get(TraceTagKey.LINKED_PROMPTS)
            else:
                for tag in run.data.tags:
                    if tag.key == TraceTagKey.LINKED_PROMPTS:
                        current_tag_value = tag.value
                        break

            updated_tag_value = update_linked_prompts_tag(current_tag_value, [prompt_version])

            if current_tag_value != updated_tag_value:
                from mlflow.entities import RunTag

                tracking_store.set_tag(
                    run_id, RunTag(TraceTagKey.LINKED_PROMPTS, updated_tag_value)
                )

    # CRUD API for Webhook objects
    def create_webhook(
        self,
        name: str,
        url: str,
        events: list[WebhookEvent],
        description: str | None = None,
        secret: str | None = None,
        status: WebhookStatus | None = None,
    ) -> Webhook:
        """
        Create a new webhook in the backend store.

        Args:
            name: Unique name for the webhook.
            url: Webhook endpoint URL.
            events: List of event types that trigger this webhook.
            description: Optional description of the webhook.
            secret: Optional secret for HMAC signature verification.
            status: Webhook status (defaults to ACTIVE).

        Returns:
            A single :py:class:`mlflow.entities.model_registry.Webhook` object
            created in the backend.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support create_webhook")

    def get_webhook(self, webhook_id: str) -> Webhook:
        """
        Get webhook instance by ID.

        Args:
            webhook_id: Webhook ID.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.Webhook` object.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support get_webhook")

    def list_webhooks(
        self,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Webhook]:
        """
        List webhooks in the backend store.

        Args:
            max_results: Maximum number of webhooks to return.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of Webhook objects.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support list_webhooks")

    def list_webhooks_by_event(
        self,
        event: WebhookEvent,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Webhook]:
        """
        List webhooks filtered by event type.

        Args:
            event: The webhook event to filter by.
            max_results: Maximum number of webhooks to return.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of Webhook objects
            that are subscribed to the specified event.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support list_webhooks_by_event"
        )

    def update_webhook(
        self,
        webhook_id: str,
        name: str | None = None,
        description: str | None = None,
        url: str | None = None,
        events: list[WebhookEvent] | None = None,
        secret: str | None = None,
        status: WebhookStatus | None = None,
    ) -> Webhook:
        """
        Update an existing webhook.

        Args:
            webhook_id: Webhook ID.
            name: New webhook name.
            description: New webhook description.
            url: New webhook URL.
            events: New list of event types.
            secret: New webhook secret.
            status: New webhook status.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.Webhook` object.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support update_webhook")

    def delete_webhook(self, webhook_id: str) -> None:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID.

        Returns:
            None
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support delete_webhook")

    def test_webhook(self, webhook_id: str, event: WebhookEvent | None = None) -> WebhookTestResult:
        """
        Test a webhook by sending a test event to the specified URL.

        Args:
            webhook_id: Webhook ID.
            event: Optional event type to test. If not specified, uses the first event from webhook.

        Returns:
            WebhookTestResult indicating success/failure and response details
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support test_webhook")
