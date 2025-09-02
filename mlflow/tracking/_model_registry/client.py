"""
Internal package providing a Python CRUD interface to MLflow models and versions.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed in the :py:mod:`mlflow.tracking` module.
"""

import logging
from typing import Any

from pydantic import BaseModel

from mlflow.entities.model_registry import (
    ModelVersionTag,
    Prompt,
    PromptVersion,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.webhook import (
    Webhook,
    WebhookEvent,
    WebhookEventStr,
    WebhookStatus,
    WebhookTestResult,
)
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import (
    add_prompt_filter_string,
    is_prompt_supported_registry,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.telemetry.events import (
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateWebhookEvent,
)
from mlflow.telemetry.track import record_usage_event
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.annotations import experimental
from mlflow.utils.arguments_utils import _get_arg_names

_logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """
    Client of an MLflow Model Registry Server that creates and manages registered
    models and model versions.
    """

    def __init__(self, registry_uri, tracking_uri):
        """
        Args:
            registry_uri: Address of local or remote model registry server.
            tracking_uri: Address of local or remote tracking server.
        """
        self.registry_uri = registry_uri
        self.tracking_uri = tracking_uri
        # NB: Fetch the tracking store (`self.store`) upon client initialization to ensure that
        # the tracking URI is valid and the store can be properly resolved. We define `store` as a
        # property method to ensure that the client is serializable, even if the store is not
        self.store

    @property
    def store(self):
        return utils._get_store(self.registry_uri, self.tracking_uri)

    # Registered Model Methods

    @record_usage_event(CreateRegisteredModelEvent)
    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
            description: Description of the model.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created by backend.

        """
        # TODO: Do we want to validate the name is legit here - non-empty without "/" and ":" ?
        #       Those are constraints applicable to any backend, given the model URI format.
        tags = tags if tags else {}
        tags = [RegisteredModelTag(key, str(value)) for key, value in tags.items()]
        return self.store.create_registered_model(name, tags, description, deployment_job_id)

    def update_registered_model(self, name, description, deployment_job_id=None):
        """Updates description for RegisteredModel entity.

        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to update.
            description: New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        return self.store.update_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )

    def rename_registered_model(self, name, new_name):
        """Update registered model name.

        Args:
            name: Name of the registered model to update.
            new_name: New proposed name for the registered model.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        if new_name.strip() == "":
            raise MlflowException("The name must not be an empty string.")
        return self.store.rename_registered_model(name=name, new_name=new_name)

    def delete_registered_model(self, name):
        """Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to delete.
        """
        self.store.delete_registered_model(name)

    def search_registered_models(
        self,
        filter_string=None,
        max_results=SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """Search for registered models in backend that satisfy the filter criteria.

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
        # Add prompt filter for prompt-supported registries that also support filter_string
        # Unity Catalog supports prompts but not filter_string parameter
        if is_prompt_supported_registry(self.registry_uri) and not (
            self.registry_uri or ""
        ).startswith("databricks-uc"):
            # Adjust filter string to include or exclude prompts
            filter_string = add_prompt_filter_string(filter_string, False)

        return self.store.search_registered_models(filter_string, max_results, order_by, page_token)

    def get_registered_model(self, name):
        """
        Args:
            name: Name of the registered model to get.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        return self.store.get_registered_model(name)

    def get_latest_versions(self, name, stages=None):
        """Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        Args:
            name: Name of the registered model from which to get the latest versions.
            stages: List of desired stages. If input list is None, return latest versions for
                'Staging' and 'Production' stages.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        """
        return self.store.get_latest_versions(name, stages)

    def set_registered_model_tag(self, name, key, value):
        """Set a tag for the registered model.

        Args:
            name: Registered model name.
            key: Tag key to log.
            value: Tag value log.

        Returns:
            None
        """
        self.store.set_registered_model_tag(name, RegisteredModelTag(key, str(value)))

    def delete_registered_model_tag(self, name, key):
        """Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        self.store.delete_registered_model_tag(name, key)

    # Model Version Methods

    @record_usage_event(CreateModelVersionEvent)
    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        local_model_path=None,
        model_id: str | None = None,
    ):
        """Create a new model version from given source.

        Args:
            name: Name of the containing registered model.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            await_creation_for: Number of seconds to wait for the model version to finish being
                created and is in ``READY`` status. By default, the function
                waits for five minutes. Specify 0 or None to skip waiting.
            local_model_path: Local path to the MLflow model, if it's already accessible on the
                local filesystem. Can be used by AbstractStores that upload model version files
                to the model registry to avoid a redundant download from the source location when
                logging and registering a model via a single
                mlflow.<flavor>.log_model(..., registered_model_name) call.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                      registered model version, if applicable.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
            backend.

        """
        tags = tags if tags else {}
        tags = [ModelVersionTag(key, str(value)) for key, value in tags.items()]
        arg_names = _get_arg_names(self.store.create_model_version)
        if "local_model_path" in arg_names:
            mv = self.store.create_model_version(
                name,
                source,
                run_id,
                tags,
                run_link,
                description,
                local_model_path=local_model_path,
                model_id=model_id,
            )
        else:
            # Fall back to calling create_model_version without
            # local_model_path since old model registry store implementations may not
            # support the local_model_path argument.
            mv = self.store.create_model_version(
                name, source, run_id, tags, run_link, description, model_id=model_id
            )
        if await_creation_for and await_creation_for > 0:
            self.store._await_model_version_creation(mv, await_creation_for)
        return mv

    def copy_model_version(self, src_mv, dst_name):
        """Copy a model version from one registered model to another as a new model version.

        Args:
            src_mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                the source model version.
            dst_name: The name of the registered model to copy the model version to. If a
                registered model with this name does not exist, it will be created.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
            the cloned model version.

        """
        return self.store.copy_model_version(src_mv=src_mv, dst_name=dst_name)

    def update_model_version(self, name, version, description):
        """Update metadata associated with a model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.
            description: New description.
        """
        return self.store.update_model_version(name=name, version=version, description=description)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        """Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be
                raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        if stage.strip() == "":
            raise MlflowException("The stage must not be an empty string.")
        return self.store.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def get_model_version(self, name, version):
        """
        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        return self.store.get_model_version(name, version)

    def delete_model_version(self, name, version):
        """Delete model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        """
        self.store.delete_model_version(name, version)

    def get_model_version_download_uri(self, name, version):
        """Get the download location in Model Registry for this model version.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        Returns:
            A single URI location that allows reads for downloading.

        """
        return self.store.get_model_version_download_uri(name, version)

    def search_model_versions(
        self,
        filter_string=None,
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """Search for model versions in backend that satisfy the filter criteria.

        .. warning:

            The model version search results may not have aliases populated for performance reasons.

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
        return self.store.search_model_versions(filter_string, max_results, order_by, page_token)

    def get_model_version_stages(self, name, version):
        """
        Returns:
            A list of valid stages.
        """
        return self.store.get_model_version_stages(name, version)

    def set_model_version_tag(self, name, version, key, value):
        """Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key to log.
            value: Tag value to log.

        Returns:
            None
        """
        self.store.set_model_version_tag(name, version, ModelVersionTag(key, str(value)))

    def delete_model_version_tag(self, name, version, key):
        """Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        self.store.delete_model_version_tag(name, version, key)

    def set_registered_model_alias(self, name, alias, version):
        """Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """
        self.store.set_registered_model_alias(name, alias, version)

    def delete_registered_model_alias(self, name, alias):
        """Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        self.store.delete_registered_model_alias(name, alias)

    def get_model_version_by_alias(self, name, alias):
        """Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        return self.store.get_model_version_by_alias(name, alias)

    @record_usage_event(CreatePromptEvent)
    def create_prompt(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Prompt:
        """
        Create a new prompt in the registry.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            description: Optional description of the prompt.
            tags: Optional dictionary of prompt tags.

        Returns:
            A PromptInfo object for Unity Catalog stores.
        """
        return self.store.create_prompt(name, description, tags)

    def get_prompt(self, name: str) -> Prompt | None:
        """
        Get prompt metadata by name.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Registered prompt name.

        Returns:
            A Prompt object with prompt metadata, or None if not found.
        """
        return self.store.get_prompt(name)

    def search_prompts(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in the registry.

        This method delegates directly to the store, providing Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            filter_string: Filter query string. For Unity Catalog registries, must include
                catalog and schema: "catalog = 'catalog_name' AND schema = 'schema_name'".
                For traditional registries, standard filter expressions are supported.
            max_results: Maximum number of prompts to return.
            order_by: List of column names with ASC|DESC annotation.
            page_token: Token specifying the next page of results.

        Returns:
            A PagedList of Prompt objects.
        """
        return self.store.search_prompts(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from the registry.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt to delete.

        Returns:
            None
        """
        self.store.delete_prompt(name)

    def create_prompt_version(
        self,
        name: str,
        template: str | list[dict[str, Any]],
        description: str | None = None,
        tags: dict[str, str] | None = None,
        response_format: BaseModel | dict[str, Any] | None = None,
    ) -> PromptVersion:
        """
        Create a new version of an existing prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            template: The prompt template content for this version. Can be either:
                - A string containing text with variables enclosed in double curly braces,
                  e.g. {{variable}}, which will be replaced with actual values by the `format`
                  method.
                - A list of dictionaries representing chat messages, where each message has
                  'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello {{name}}"}])
            description: Optional description of this version.
            tags: Optional dictionary of version tags.
            response_format: Optional Pydantic class or dictionary defining the expected response
                structure. This can be used to specify the schema for structured outputs from LLM
                calls.

        Returns:
            A PromptVersion object representing the new version.
        """
        return self.store.create_prompt_version(name, template, description, tags, response_format)

    def get_prompt_version(self, name: str, version: str) -> PromptVersion:
        """
        Get a specific version of a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version number of the prompt.

        Returns:
            A PromptVersion object.
        """
        return self.store.get_prompt_version(name, version)

    def delete_prompt_version(self, name: str, version: str) -> None:
        """
        Delete a specific version of a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version number to delete.

        Returns:
            None
        """
        self.store.delete_prompt_version(name, version)

    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            key: Tag key.
            value: Tag value.

        Returns:
            None
        """
        self.store.set_prompt_tag(name, key, value)

    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            key: Tag key to delete.

        Returns:
            None
        """
        self.store.delete_prompt_tag(name, key)

    def get_prompt_version_by_alias(self, name: str, alias: str) -> PromptVersion:
        """
        Get a prompt version by alias.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            alias: Alias to look up.

        Returns:
            A PromptVersion object.
        """
        return self.store.get_prompt_version_by_alias(name, alias)

    def set_prompt_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set an alias for a prompt version.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            alias: Alias to set.
            version: Version to alias.

        Returns:
            None
        """
        self.store.set_prompt_alias(name, alias, version)

    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete a prompt alias.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            alias: Alias to delete.

        Returns:
            None
        """
        self.store.delete_prompt_alias(name, alias)

    def search_prompt_versions(
        self, name: str, max_results: int | None = None, page_token: str | None = None
    ):
        """
        Search prompt versions for a given prompt name.

        This method delegates directly to the store. Only supported in Unity Catalog registries.

        Args:
            name: Name of the prompt to search versions for.
            max_results: Maximum number of versions to return.
            page_token: Token for pagination.

        Returns:
            SearchPromptVersionsResponse containing the list of versions.

        Raises:
            MlflowException: If used with non-Unity Catalog registries.
        """
        return self.store.search_prompt_versions(name, max_results, page_token)

    def link_prompt_version_to_model(self, name: str, version: int | str, model_id: str) -> None:
        """
        Link a prompt version to a model.

        Args:
            name: The name of the prompt.
            version: The version of the prompt.
            model_id: The ID of the model to link the prompt version to.
        """
        return self.store.link_prompt_version_to_model(name, str(version), model_id)

    def link_prompt_version_to_run(self, name: str, version: int | str, run_id: str) -> None:
        """
        Link a prompt version to a run.

        Args:
            name: The name of the prompt.
            version: The version of the prompt.
            run_id: The ID of the run to link the prompt version to.
        """
        return self.store.link_prompt_version_to_run(name, str(version), run_id)

    def link_prompt_versions_to_trace(
        self, prompt_versions: list[PromptVersion], trace_id: str
    ) -> None:
        """
        Link multiple prompt versions to a trace.

        Args:
            prompt_versions: List of PromptVersion objects to link.
            trace_id: Trace ID to link the prompt versions to.
        """
        return self.store.link_prompts_to_trace(prompt_versions=prompt_versions, trace_id=trace_id)

    def set_prompt_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt version.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version number of the prompt.
            key: Tag key.
            value: Tag value.

        Returns:
            None
        """
        self.store.set_prompt_version_tag(name, version, key, value)

    def delete_prompt_version_tag(self, name: str, version: str, key: str) -> None:
        """
        Delete a tag from a prompt version.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version number of the prompt.
            key: Tag key to delete.

        Returns:
            None
        """
        self.store.delete_prompt_version_tag(name, version, key)

    # Webhook APIs
    @experimental(version="3.3.0")
    @record_usage_event(CreateWebhookEvent)
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
        Create a new webhook.

        Args:
            name: Unique name for the webhook.
            url: Webhook endpoint URL.
            events: List of event types that trigger this webhook.
            description: Optional description of the webhook.
            secret: Optional secret for HMAC signature verification.
            status: Webhook status (defaults to ACTIVE).

        Returns:
            A :py:class:`mlflow.entities.webhook.Webhook` object representing the created webhook.
        """
        return self.store.create_webhook(name, url, events, description, secret, status)

    @experimental(version="3.3.0")
    def get_webhook(self, webhook_id: str) -> Webhook:
        """
        Get webhook instance by ID.

        Args:
            webhook_id: Webhook ID.

        Returns:
            A :py:class:`mlflow.entities.webhook.Webhook` object.
        """
        return self.store.get_webhook(webhook_id)

    @experimental(version="3.3.0")
    def list_webhooks(
        self,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Webhook]:
        """
        List webhooks.

        Args:
            max_results: Maximum number of webhooks to return.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`mlflow.store.entities.paged_list.PagedList` of Webhook objects.
        """
        return self.store.list_webhooks(max_results, page_token)

    @experimental(version="3.3.0")
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
            A :py:class:`mlflow.entities.webhook.Webhook` object representing the updated webhook.
        """
        return self.store.update_webhook(webhook_id, name, description, url, events, secret, status)

    @experimental(version="3.3.0")
    def delete_webhook(self, webhook_id: str) -> None:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID to delete.

        Returns:
            None
        """
        self.store.delete_webhook(webhook_id)

    @experimental(version="3.3.0")
    def test_webhook(
        self, webhook_id: str, event: WebhookEventStr | WebhookEvent | None = None
    ) -> WebhookTestResult:
        """
        Test a webhook by sending a test payload.

        Args:
            webhook_id: The ID of the webhook to test.
            event: Optional event type to test. Can be a WebhookEvent object or a string in
                "entity.action" format (e.g., "model_version.created"). If not specified, uses
                the first event from webhook.

        Returns:
            A :py:class:`mlflow.entities.webhook.WebhookTestResult` indicating success/failure and
            response details.
        """
        # Convert string to WebhookEvent if needed
        if isinstance(event, str):
            event = WebhookEvent.from_str(event)
        return self.store.test_webhook(webhook_id, event)
