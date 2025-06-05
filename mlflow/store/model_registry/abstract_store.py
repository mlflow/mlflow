import logging
from abc import ABCMeta, abstractmethod
from time import sleep, time
from typing import Optional, Union

from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY
from mlflow.prompt.registry_utils import has_prompt_tag
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store._unity_catalog.registry.prompt_info import PromptInfo
from mlflow.store.entities.paged_list import PagedList
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
        model_id: Optional[str] = None,
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
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> PromptInfo:
        """
        Create a new prompt in the registry.

        Default implementation: creates a RegisteredModel with special prompt tags.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            description: Optional description of the prompt.
            tags: Optional dictionary of prompt tags.

        Returns:
            A PromptInfo object representing the created prompt.
        """
        # Default implementation: use RegisteredModel with special tags
        prompt_tags = [RegisteredModelTag(key=IS_PROMPT_TAG_KEY, value="true")]
        if tags:
            prompt_tags.extend([RegisteredModelTag(key=k, value=v) for k, v in tags.items()])

        # Create registered model for the prompt
        rm = self.create_registered_model(name, tags=prompt_tags, description=description)

        # Return as PromptInfo
        return PromptInfo(
            name=rm.name,
            description=rm.description,
            creation_timestamp=rm.creation_timestamp,
            tags=tags or {},
        )

    def search_prompts(
        self,
        filter_string: Optional[str] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[PromptInfo]:
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
            A PagedList of PromptInfo objects.
        """
        if max_results is None:
            max_results = 100

        # Build filter to only include prompts (use backticks for tag key with dots)
        prompt_filter = f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'"
        if filter_string:
            prompt_filter = f"{prompt_filter} AND {filter_string}"

        # Search registered models with prompt filter
        registered_models = self.search_registered_models(
            filter_string=prompt_filter,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

        # Convert RegisteredModel objects to PromptInfo objects
        prompts = []
        for rm in registered_models:
            # Extract tags as dict
            if isinstance(rm.tags, dict):
                tags = rm.tags.copy()
            else:
                tags = {tag.key: tag.value for tag in rm.tags} if rm.tags else {}

            # Remove the internal prompt tag from user-visible tags
            tags.pop(IS_PROMPT_TAG_KEY, None)

            # Create PromptInfo object
            prompt_info = PromptInfo(
                name=rm.name,
                description=rm.description,
                creation_timestamp=rm.creation_timestamp,
                last_updated_timestamp=rm.last_updated_timestamp,
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

    def get_prompt(self, name: str, version: Optional[Union[str, int]] = None) -> Optional[Prompt]:
        """
        Get prompt by name and version or alias.

        Default implementation: gets ModelVersion with prompt tags and converts to Prompt.
        Other store implementations may override this method.

        Args:
            name: Registered prompt name.
            version: Registered prompt version or alias. If None, loads the latest version.

        Returns:
            A single Prompt object, or None if not found.
        """
        if version is None:
            latest_versions = self.get_latest_versions(name, stages=ALL_STAGES)
            if not latest_versions:
                return None
            mv = latest_versions[0]
        else:
            version_int = int(str(version))
            mv = self.get_model_version(name, version_int)

        if not has_prompt_tag(mv.tags):
            return None

        # Get prompt-level tags from registered model
        rm = self.get_registered_model(name)
        if isinstance(rm.tags, dict):
            prompt_tags = rm.tags.copy()
        else:
            prompt_tags = {tag.key: tag.value for tag in rm.tags}

        return Prompt.from_model_version(mv, prompt_tags=prompt_tags)

    def create_prompt_version(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Prompt:
        """
        Create a new version of an existing prompt.

        Default implementation: creates a ModelVersion with prompt tags.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            template: The prompt template text.
            description: Optional description of the prompt version.
            tags: Optional dictionary of version tags.

        Returns:
            A Prompt object representing the created version.
        """
        # Create version tags including template
        version_tags = [
            ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
            ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value=template),
        ]
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

        return Prompt.from_model_version(mv, prompt_tags=prompt_tags)

    def get_prompt_version(self, name: str, version: Union[str, int]) -> Optional[Prompt]:
        """
        Get a specific prompt version.

        Default implementation: gets ModelVersion and converts to Prompt.
        Other store implementations may override this method.

        Args:
            name: Name of the prompt.
            version: Version number or alias.

        Returns:
            A Prompt object, or None if not found.
        """
        return self.get_prompt(name, version)

    def delete_prompt_version(self, name: str, version: Union[str, int]) -> None:
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

    def get_prompt_version_by_alias(self, name: str, alias: str) -> Optional[Prompt]:
        """
        Get a prompt version by alias.

        Default implementation: uses get_model_version_by_alias and converts to Prompt.

        Args:
            name: Name of the prompt.
            alias: Alias name.

        Returns:
            A Prompt object, or None if not found.
        """
        return self.get_prompt(name, alias)

    def set_prompt_alias(self, name: str, alias: str, version: Union[str, int]) -> None:
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
