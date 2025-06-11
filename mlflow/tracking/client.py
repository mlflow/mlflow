"""
Internal package providing a Python CRUD interface to MLflow experiments, runs, registered models,
and model versions. This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module,
and is exposed in the :py:mod:`mlflow.tracking` module.
"""

import contextlib
import json
import logging
import os
import posixpath
import random
import re
import string
import sys
import tempfile
import urllib
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import yaml

import mlflow
from mlflow.entities import (
    DatasetInput,
    Experiment,
    FileInfo,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelStatus,
    Metric,
    Param,
    Run,
    RunTag,
    Span,
    SpanStatus,
    SpanType,
    Trace,
    ViewType,
)
from mlflow.entities.assessment import (
    Assessment,
    Expectation,
    Feedback,
)
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.model_registry import ModelVersion, Prompt, PromptVersion, RegisteredModel
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID, NoOpSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
)
from mlflow.prompt.registry_utils import (
    has_prompt_tag,
    model_version_to_prompt_version,
    parse_prompt_name_or_uri,
    require_prompt_registry,
    translate_prompt_exception,
    validate_prompt_name,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    FEATURE_DISABLED,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TRACE_REQUEST_ID_PREFIX
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils.warning import request_id_backward_compatible
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.client import ModelRegistryClient
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.tracking.artifact_utils import _upload_artifacts_to_databricks
from mlflow.tracking.multimedia import Image, compress_image_size, convert_to_pil_image
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils import is_uuid
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.databricks_utils import (
    get_databricks_run_url,
    is_in_databricks_runtime,
)
from mlflow.utils.logging_utils import eprint
from mlflow.utils.mlflow_tags import (
    MLFLOW_LOGGED_ARTIFACTS,
    MLFLOW_LOGGED_IMAGES,
    MLFLOW_PARENT_RUN_ID,
)
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import is_databricks_unity_catalog_uri, is_databricks_uri
from mlflow.utils.validation import (
    _validate_model_alias_name,
    _validate_model_name,
    _validate_model_version,
    _validate_model_version_or_stage_exists,
)

if TYPE_CHECKING:
    import matplotlib
    import numpy
    import pandas
    import PIL
    import plotly


_logger = logging.getLogger(__name__)

_STAGES_DEPRECATION_WARNING = (
    "Model registry stages will be removed in a future major release. To learn more about the "
    "deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/"
    "latest/model-registry.html#migrating-from-stages"
)


def _model_not_found(name: str) -> MlflowException:
    return MlflowException(
        f"Registered Model with name={name!r} not found.",
        RESOURCE_DOES_NOT_EXIST,
    )


def _validate_model_id_specified(model_id: str) -> None:
    if not model_id:
        raise MlflowException(
            f"`model_id` must be a non-empty string, but got {model_id!r}",
            INVALID_PARAMETER_VALUE,
        )


class MlflowClient:
    """
    Client of an MLflow Tracking Server that creates and manages experiments and runs, and of an
    MLflow Registry Server that creates and manages registered models and model versions. It's a
    thin wrapper around TrackingServiceClient and RegistryClient so there is a unified API but we
    can keep the implementation of the tracking and registry clients independent from each other.
    """

    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """
        Args:
            tracking_uri: Address of local or remote tracking server. If not provided, defaults
                to the service set by ``mlflow.tracking.set_tracking_uri``. See
                `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                for more info.
            registry_uri: Address of local or remote model registry server. If not provided,
                defaults to the service set by ``mlflow.tracking.set_registry_uri``. If
                no such service was set, defaults to the tracking uri of the client.
        """
        final_tracking_uri = utils._resolve_tracking_uri(tracking_uri)
        self._registry_uri = registry_utils._resolve_registry_uri(registry_uri, tracking_uri)
        self._tracking_client = TrackingServiceClient(final_tracking_uri)
        self._tracing_client = TracingClient(tracking_uri)

        # `MlflowClient` also references a `ModelRegistryClient` instance that is provided by the
        # `MlflowClient._get_registry_client()` method. This `ModelRegistryClient` is not explicitly
        # defined as an instance variable in the `MlflowClient` constructor; an instance variable
        # is assigned lazily by `MlflowClient._get_registry_client()` and should not be referenced
        # outside of the `MlflowClient._get_registry_client()` method

    @property
    def tracking_uri(self):
        return self._tracking_client.tracking_uri

    def _get_registry_client(self):
        """Attempts to create a ModelRegistryClient if one does not already exist.

        Raises:
            MlflowException: If the ModelRegistryClient cannot be created. This may occur, for
            example, when the registry URI refers to an unsupported store type (e.g., the
            FileStore).

        Returns:
            A ModelRegistryClient instance.
        """
        # Attempt to fetch a `ModelRegistryClient` that is lazily instantiated and defined as
        # an instance variable on this `MlflowClient` instance. Because the instance variable
        # is undefined until the first invocation of _get_registry_client(), the `getattr()`
        # function is used to safely fetch the variable (if it is defined) or a NoneType
        # (if it is not defined)
        registry_client_attr = "_registry_client_lazy"
        registry_client = getattr(self, registry_client_attr, None)
        if registry_client is None:
            try:
                registry_client = ModelRegistryClient(self._registry_uri, self.tracking_uri)
                # Define an instance variable on this `MlflowClient` instance to reference the
                # `ModelRegistryClient` that was just constructed. `setattr()` is used to ensure
                # that the variable name is consistent with the variable name specified in the
                # preceding call to `getattr()`
                setattr(self, registry_client_attr, registry_client)
            except UnsupportedModelRegistryStoreURIException as exc:
                raise MlflowException(
                    "Model Registry features are not supported by the store with URI:"
                    f" '{self._registry_uri}'. Stores with the following URI schemes are supported:"
                    f" {exc.supported_uri_schemes}.",
                    FEATURE_DISABLED,
                )
        return registry_client

    # Tracking API

    def get_run(self, run_id: str) -> Run:
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata -- :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics --
        :py:class:`RunData <mlflow.entities.RunData>`. It also contains a collection of run
        inputs (experimental), including information about datasets used by the run --
        :py:class:`RunInputs <mlflow.entities.RunInputs>`. In the case where multiple metrics with
        the same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>`
        contains the most recently logged value at the largest step for each metric.

        Args:
            run_id: Unique identifier for the run.

        Returns:
            A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
            raises an exception.


        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            with mlflow.start_run() as run:
                mlflow.log_param("p", 0)

            # The run has finished since we have exited the with block
            # Fetch the run
            client = MlflowClient()
            run = client.get_run(run.info.run_id)
            print(f"run_id: {run.info.run_id}")
            print(f"params: {run.data.params}")
            print(f"status: {run.info.status}")

        .. code-block:: text
            :caption: Output

            run_id: e36b42c587a1413ead7c3b6764120618
            params: {'p': '0'}
            status: FINISHED

        """
        return self._tracking_client.get_run(run_id)

    def get_parent_run(self, run_id: str) -> Optional[Run]:
        """Gets the parent run for the given run id if one exists.

        Args:
            run_id: Unique identifier for the child run.

        Returns:
            A single :py:class:`mlflow.entities.Run` object, if the parent run exists. Otherwise,
            returns None.

        .. code-block:: python
            :test:
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            # Create nested runs
            with mlflow.start_run():
                with mlflow.start_run(nested=True) as child_run:
                    child_run_id = child_run.info.run_id

            client = MlflowClient()
            parent_run = client.get_parent_run(child_run_id)

            print(f"child_run_id: {child_run_id}")
            print(f"parent_run_id: {parent_run.info.run_id}")

        .. code-block:: text
            :caption: Output

            child_run_id: 7d175204675e40328e46d9a6a5a7ee6a
            parent_run_id: 8979459433a24a52ab3be87a229a9cdf

        """
        child_run = self._tracking_client.get_run(run_id)
        parent_run_id = child_run.data.tags.get(MLFLOW_PARENT_RUN_ID)
        if parent_run_id is None:
            return None
        return self._tracking_client.get_run(parent_run_id)

    def get_metric_history(self, run_id: str, key: str) -> list[Metric]:
        """Return a list of metric objects corresponding to all values logged for a given metric.

        Args:
            run_id: Unique identifier for run.
            key: Metric name within the run.

        Returns:
            A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_metric_info(history):
                for m in history:
                    print(f"name: {m.key}")
                    print(f"value: {m.value}")
                    print(f"step: {m.step}")
                    print(f"timestamp: {m.timestamp}")
                    print("--")


            # Create a run under the default experiment (whose id is "0"). Since this is low-level
            # CRUD operation, the method will create a run. To end the run, you'll have
            # to explicitly end it.
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print(f"run_id: {run.info.run_id}")
            print("--")

            # Log couple of metrics, update their initial value, and fetch each
            # logged metrics' history.
            for k, v in [("m1", 1.5), ("m2", 2.5)]:
                client.log_metric(run.info.run_id, k, v, step=0)
                client.log_metric(run.info.run_id, k, v + 1, step=1)
                print_metric_info(client.get_metric_history(run.info.run_id, k))
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            run_id: c360d15714994c388b504fe09ea3c234
            --
            name: m1
            value: 1.5
            step: 0
            timestamp: 1603423788607
            --
            name: m1
            value: 2.5
            step: 1
            timestamp: 1603423788608
            --
            name: m2
            value: 2.5
            step: 0
            timestamp: 1603423788609
            --
            name: m2
            value: 3.5
            step: 1
            timestamp: 1603423788610
            --
        """
        return self._tracking_client.get_metric_history(run_id, key)

    def create_run(
        self,
        experiment_id: str,
        start_time: Optional[int] = None,
        tags: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> Run:
        """
        Create a :py:class:`mlflow.entities.Run` object that can be associated with
        metrics, parameters, artifacts, etc.
        Unlike :py:func:`mlflow.projects.run`, creates objects but does not run code.
        Unlike :py:func:`mlflow.start_run`, does not change the "active run" used by
        :py:func:`mlflow.log_param`.

        Args:
            experiment_id: The string ID of the experiment to create a run in.
            start_time: If not provided, use the current timestamp.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.RunTag` objects.
            run_name: The name of this run.

        Returns:
            :py:class:`mlflow.entities.Run` that was created.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create a run with a tag under the default experiment (whose id is '0').
            tags = {"engineering": "ML Platform"}
            name = "platform-run-24"
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id, tags=tags, run_name=name)

            # Show newly created run metadata info
            print(f"Run tags: {run.data.tags}")
            print(f"Experiment id: {run.info.experiment_id}")
            print(f"Run id: {run.info.run_id}")
            print(f"Run name: {run.info.run_name}")
            print(f"lifecycle_stage: {run.info.lifecycle_stage}")
            print(f"status: {run.info.status}")

        .. code-block:: text
            :caption: Output

            Run tags: {'engineering': 'ML Platform'}
            Experiment id: 0
            Run id: 65fb9e2198764354bab398105f2e70c1
            Run name: platform-run-24
            lifecycle_stage: active
            status: RUNNING
        """
        return self._tracking_client.create_run(experiment_id, start_time, tags, run_name)

    ##### Prompt Registry #####

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> PromptVersion:
        """
        Register a new :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

        A :py:class:`Prompt <mlflow.entities.Prompt>` is a pair of name
        and template text at minimum. With MLflow Prompt Registry, you can create, manage,
        and version control prompts with the MLflow's robust model tracking framework.

        If there is no registered prompt with the given name, a new prompt will be created.
        Otherwise, a new version of the existing prompt will be created.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            # Your prompt registry URI
            client = MlflowClient(registry_uri="sqlite:///prompt_registry.db")

            # Register a new prompt
            client.register_prompt(
                name="my_prompt",
                template="Respond to the user's message as a {{style}} AI.",
            )

            # Load the prompt from the registry
            prompt = client.load_prompt("my_prompt")

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
            prompt = client.register_prompt(
                name="my_prompt",
                template="Respond to the user's message as a {{style}} AI. {{greeting}}",
                commit_message="Add a greeting to the prompt.",
                tags={"author": "Bob"},
            )

        Args:
            name: The name of the prompt.
            template: The template text of the prompt. It can contain variables enclosed in
                double curly braces, e.g. {{variable}}, which will be replaced with actual values
                by the `format` method.
            commit_message: A message describing the changes made to the prompt, similar to a
                Git commit message. Optional.
            tags: A dictionary of tags associated with the **prompt version**.
                This is useful for storing version-specific information, such as the author of
                the changes. Optional.

        Returns:
            A :py:class:`Prompt <mlflow.entities.Prompt>` object that was created.
        """
        registry_client = self._get_registry_client()

        validate_prompt_name(name)

        if is_databricks_unity_catalog_uri(self._registry_uri):
            try:
                registry_client.create_prompt(
                    name=name, description=commit_message, tags=tags or {}
                )
            except MlflowException as e:
                if e.error_code == ErrorCode.Name(ErrorCode.ALREADY_EXISTS):
                    pass
                else:
                    # Re-raise other errors like permission issues, validation errors, etc.
                    raise

            # Create the prompt version
            prompt_version = registry_client.create_prompt_version(
                name=name,
                template=template,
                description=commit_message,
                tags=tags or {},
            )

            return registry_client.get_prompt_version(name, str(prompt_version.version))

        # OSS approach using RegisteredModel with special tags
        is_new_prompt = False
        rm = None
        tags = tags or {}
        try:
            rm = registry_client.get_registered_model(name)
        except MlflowException:
            # Create a new prompt (model) entry
            registry_client.create_registered_model(
                name, description=commit_message, tags={IS_PROMPT_TAG_KEY: "true", **tags}
            )
            is_new_prompt = True

        # Check if the registered model is a prompt
        if rm and not has_prompt_tag(rm._tags):
            raise MlflowException(
                f"Model '{name}' exists with the same name. MLflow does not allow registering "
                "a prompt with the same name as an existing model. Please choose a different "
                "name for the prompt.",
                INVALID_PARAMETER_VALUE,
            )

        # Update the prompt tags
        if not is_new_prompt:
            for key, value in tags.items():
                registry_client.set_registered_model_tag(name, key, value)

        # Version metadata is represented as ModelVersion tags in the registry
        tags = tags or {}
        tags.update({IS_PROMPT_TAG_KEY: "true", PROMPT_TEXT_TAG_KEY: template})

        try:
            mv: ModelVersion = registry_client.create_model_version(
                name=name,
                description=commit_message,
                source="dummy-source",  # Required field, but not used for prompts
                tags=tags,
            )
        except Exception:
            if is_new_prompt:
                # When a model version creation fails for the first version of a prompt,
                # delete the registered model to avoid leaving a prompt with no versions
                registry_client.delete_registered_model(name)
            raise

        # Fetch the prompt-level tags from the registered model
        prompt_tags = registry_client.get_registered_model(name)._tags

        return model_version_to_prompt_version(mv, prompt_tags=prompt_tags)

    @translate_prompt_exception
    @require_prompt_registry
    def search_prompts(
        self,
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        page_token: Optional[str] = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in the MLflow Prompt Registry.

        This call returns prompt metadata for prompts that have been marked
        as prompts (i.e. tagged with `mlflow.prompt.is_prompt=true`). We can
        further restrict results via a standard registry filter expression.

        Args:
            filter_string (Optional[str]):
                An additional registry‐search expression to apply (e.g.
                `"name LIKE 'my_prompt%'"`).  For Unity Catalog registries, must include
                catalog and schema: "catalog = 'catalog_name' AND schema = 'schema_name'".
            max_results (int):
                The maximum number of prompts to return in one page.  Defaults
                to `SEARCH_MAX_RESULTS_DEFAULT` (typically 1 000).
            page_token (Optional[str]):
                A pagination token from a previous `search_prompts` call; use this
                to retrieve the next page of results.  Defaults to `None`.

        Returns:
            A pageable list of Prompt objects representing prompt metadata:

            - name: The prompt name
            - description: The prompt description
            - tags: Prompt-level tags
            - creation_timestamp: When the prompt was created

            To get the actual prompt template content, use get_prompt() with a specific version:

            .. code-block:: python

                # Search for prompts
                prompts = client.search_prompts(filter_string="name LIKE 'greeting%'")

                # Get specific version content
                for prompt in prompts:
                    prompt_version = client.get_prompt_version(prompt.name, version="1")
                    print(f"Template: {prompt.template}")

            Inspect the returned object's `.token` attribute to fetch subsequent pages.
        """
        registry_client = self._get_registry_client()

        # Delegate to the store - each store handles its own implementation
        return registry_client.search_prompts(
            filter_string=filter_string,
            max_results=max_results,
            page_token=page_token,
        )

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def load_prompt(
        self,
        name_or_uri: str,
        version: Optional[Union[str, int]] = None,
        allow_missing: bool = False,
    ) -> Optional[PromptVersion]:
        """
        Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

        The prompt can be specified by name and version, or by URI.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient(registry_uri="sqlite:///prompt_registry.db")

            # Load a specific version of the prompt by name and version
            prompt = client.load_prompt("my_prompt", version=1)

            # Load a specific version of the prompt by URI
            prompt = client.load_prompt("prompts:/my_prompt/1")

        Args:
            name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
            version: The version of the prompt (required when using name, not allowed when
                using URI).
            allow_missing: If True, return None instead of raising Exception if the specified prompt
                is not found.
        """
        parsed_name_or_uri, parsed_version = parse_prompt_name_or_uri(name_or_uri, version)
        if parsed_name_or_uri.startswith("prompts:/"):
            # URI case: parse the URI to extract name and version
            name, version = self.parse_prompt_uri(parsed_name_or_uri)
        else:
            # Name case: use the name and provided version
            name = parsed_name_or_uri
            version = parsed_version

        registry_client = self._get_registry_client()
        try:
            # Use get_prompt_version for specific version/alias
            return registry_client.get_prompt_version(name, version)

        except MlflowException as exc:
            if allow_missing and exc.error_code in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
                return None
            raise

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def link_prompt_version_to_run(self, run_id: str, prompt: Union[str, PromptVersion]) -> None:
        """
        Link a prompt registered within the MLflow Prompt Registry with an MLflow Run.

        .. warning::

            This API is not thread-safe. If you are linking prompts from multiple threads,
            consider using a lock to ensure that only one thread links a prompt to a run at a time.

        Args:
            run_id: The ID of the run to link the prompt to.
            prompt: A Prompt object or the prompt URI in the format "prompts:/name/version".
        """
        if isinstance(prompt, str):
            prompt = self.load_prompt(prompt)
        elif isinstance(prompt, PromptVersion):
            # NB: We need to load the prompt once from the registry because the tags in
            # local prompt object may not be in sync with the registry.
            prompt = self.load_prompt(prompt.uri)
        elif not isinstance(prompt, PromptVersion):
            raise MlflowException.invalid_parameter_value(
                "The `prompt` argument must be a Prompt object or a prompt URI.",
            )

        self._get_registry_client().link_prompt_version_to_run(
            name=prompt.name, version=prompt.version, run_id=run_id
        )

    def link_prompt_version_to_model(self, name: str, version: str, model_id: str) -> None:
        """
        Link a prompt version to a model.

        Args:
            name: The name of the prompt.
            version: The version of the prompt.
            model_id: The ID of the model to link the prompt version to.
        """
        return self._get_registry_client().link_prompt_version_to_model(
            name=name,
            version=version,
            model_id=model_id,
        )

    def link_prompt_versions_to_trace(
        self, prompt_versions: list[PromptVersion], trace_id: str
    ) -> None:
        """
        Link multiple prompt versions to a trace.

        Args:
            prompt_versions: List of PromptVersion objects to link.
            trace_id: Trace ID to link to each prompt version.

        Example:
            .. code-block:: python

                import mlflow
                from mlflow import MlflowClient

                client = MlflowClient()

                # Get prompt versions and link to trace
                prompt_v1 = client.get_prompt_version("my_prompt", "1")
                prompt_v2 = client.get_prompt_version("another_prompt", "2")

                client.link_prompt_versions_to_trace(
                    prompt_versions=[prompt_v1, prompt_v2],
                    trace_id="trace_123",
                )
        """
        return self._get_registry_client().link_prompt_versions_to_trace(
            prompt_versions=prompt_versions,
            trace_id=trace_id,
        )

    # TODO: Use model_id in MLflow 3.0
    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def detach_prompt_from_run(self, run_id: str, prompt_uri: str) -> None:
        """
        Detach a prompt registered within the MLflow Prompt Registry from an MLflow Run.

        Args:
            run_id: The ID of the run to log the prompt to.
            prompt_uri: The prompt URI in the format "prompts:/name/version".
        """
        prompt = self.load_prompt(prompt_uri)
        run_id_tags = prompt._tags.get(PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY)
        run_ids = run_id_tags.split(",") if run_id_tags else []

        if run_id not in run_ids:
            raise MlflowException(
                f"Run '{run_id}' is not associated with prompt '{prompt_uri}'.",
                INVALID_PARAMETER_VALUE,
            )

        run_ids.remove(run_id)

        name, version = self.parse_prompt_uri(prompt_uri)
        if run_ids:
            self._get_registry_client().set_model_version_tag(
                name, version, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, ",".join(run_ids)
            )
        else:
            self._get_registry_client().delete_model_version_tag(
                name, version, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY
            )

    # TODO: Use model_id in MLflow 3.0
    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def list_logged_prompts(self, run_id: str) -> list[PromptVersion]:
        """
        List all prompts associated with an MLflow Run.

        Args:
            run_id: The ID of the run to list the prompts for.

        Returns:
            A list of :py:class:`Prompt <mlflow.entities.Prompt>` objects associated with the run.
        """
        mvs = self.search_model_versions(
            filter_string=(
                f"tags.`{PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY}` LIKE '%{run_id}%' "
                f"AND tags.`{IS_PROMPT_TAG_KEY}` = 'true'"
            )
        )
        # NB: We don't support pagination here because the number of prompts associated
        # with a Run is expected to be small.
        return [model_version_to_prompt_version(mv) for mv in mvs]

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        """
        Set an alias for a :py:class:`Prompt <mlflow.entities.Prompt>`.

        Args:
            name: The name of the prompt.
            alias: The alias to set for the prompt.
            version: The version of the prompt.
        """
        self._validate_prompt(name, version)
        self._get_registry_client().set_prompt_alias(name, alias, version)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete an alias for a :py:class:`Prompt <mlflow.entities.Prompt>`.

        Args:
            name: The name of the prompt.
            alias: The alias to delete for the prompt.
        """
        self._get_registry_client().delete_prompt_alias(name, alias)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def set_prompt_version_tag(
        self, name: str, version: Union[str, int], key: str, value: str
    ) -> None:
        """
        Set a tag on a specific prompt version.

        Args:
            name: The name of the prompt.
            version: The version number of the prompt.
            key: The tag key.
            value: The tag value.
        """
        self._get_registry_client().set_prompt_version_tag(name, version, key, value)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def delete_prompt_version_tag(self, name: str, version: Union[str, int], key: str) -> None:
        """
        Delete a tag from a specific prompt version.

        Args:
            name: The name of the prompt.
            version: The version number of the prompt.
            key: The tag key to delete.
        """
        self._get_registry_client().delete_prompt_version_tag(name, version, key)

    def _validate_prompt(self, name: str, version: int):
        registry_client = self._get_registry_client()
        try:
            pv = registry_client.get_prompt_version(name, version)
            if pv is None:
                raise MlflowException(
                    f"Prompt '{name}' version {version} does not exist.", RESOURCE_DOES_NOT_EXIST
                )
        except Exception:
            raise MlflowException(
                f"Prompt '{name}' version {version} does not exist.", RESOURCE_DOES_NOT_EXIST
            )

    def parse_prompt_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse prompt URI into prompt name and prompt version.
        - 'prompt:/<name>/<version>' -> ('<name>', '<version>')
        - 'prompt:/<name>@<alias>' -> ('<name>', '<version>')
        """
        parsed = urllib.parse.urlparse(uri)

        if parsed.scheme != "prompts":
            raise MlflowException.invalid_parameter_value(
                f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
            )

        # Replace schema to 'models:/' to reuse the model URI parsing logic
        try:
            return get_model_name_and_version(self, f"models:{parsed.path}")
        except MlflowException:
            raise MlflowException(f"Prompt '{uri}' does not exist.", RESOURCE_DOES_NOT_EXIST)

    ##### Tracing #####
    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        trace_ids: Optional[list[str]] = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        - Either `max_timestamp_millis` or `trace_ids` must be specified, but not both.
        - `max_traces` can't be specified if `trace_ids` is specified.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete. If max_traces is specified, and
                it is less than the number of traces that would be deleted based on the
                max_timestamp_millis, the oldest traces will be deleted first.
            trace_ids: A set of trace IDs to delete.

        Returns:
            The number of traces deleted.

        Example:

        .. code-block:: python
            :test:

            import mlflow
            import time

            client = mlflow.MlflowClient()

            # Delete all traces in the experiment
            client.delete_traces(
                experiment_id="0", max_timestamp_millis=time.time_ns() // 1_000_000
            )

            # Delete traces based on max_timestamp_millis and max_traces
            # Older traces will be deleted first.
            some_timestamp = time.time_ns() // 1_000_000
            client.delete_traces(
                experiment_id="0", max_timestamp_millis=some_timestamp, max_traces=2
            )

            # Delete traces based on trace_ids
            client.delete_traces(experiment_id="0", trace_ids=["id_1", "id_2"])
        """
        return self._tracing_client.delete_traces(
            experiment_id=experiment_id,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
            request_ids=trace_ids,
        )

    @request_id_backward_compatible
    def get_trace(self, trace_id: str, display=True) -> Trace:
        """
        Get the trace matching the specified ``trace_id``.

        Args:
            trace_id: String ID of the trace to fetch.
            display: If ``True``, display the trace on the notebook.

        Returns:
            The retrieved :py:class:`Trace <mlflow.entities.Trace>`.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            client = MlflowClient()
            trace_id = "12345678"
            trace = client.get_trace(trace_id)
        """
        if is_databricks_uri(str(self.tracking_uri)) and is_uuid(trace_id):
            raise MlflowException.invalid_parameter_value(
                "Traces from inference tables can only be loaded using SQL or "
                "the search_traces() API."
            )

        trace = self._tracing_client.get_trace(trace_id)
        if display:
            get_display_handler().display_traces([trace])
        return trace

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
        run_id: Optional[str] = None,
        include_spans: bool = True,
        model_id: Optional[str] = None,
        sql_warehouse_id: Optional[str] = None,
    ) -> PagedList[Trace]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            max_results: Maximum number of traces desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_traces`` call.
            run_id: A run id to scope the search. When a trace is created under an active run,
                it will be associated with the run and you can filter on the run id to retrieve
                the trace.
            include_spans: If ``True``, include spans in the returned traces. Otherwise, only
                the trace metadata is returned, e.g., trace ID, start time, end time, etc,
                without any spans.
            model_id: If specified, return traces associated with the model ID.
            sql_warehouse_id: Only used in Databricks. The ID of the SQL warehouse to use for
                searching traces in inference tables.


        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Trace <mlflow.entities.Trace>` objects that satisfy the search
            expressions. If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        return self._tracing_client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            run_id=run_id,
            include_spans=include_spans,
            model_id=model_id,
            sql_warehouse_id=sql_warehouse_id,
        )

    def start_trace(
        self,
        name: str,
        span_type: str = SpanType.UNKNOWN,
        inputs: Optional[Any] = None,
        attributes: Optional[dict[str, str]] = None,
        tags: Optional[dict[str, str]] = None,
        experiment_id: Optional[str] = None,
        start_time_ns: Optional[int] = None,
    ) -> Span:
        """
        Create a new trace object and start a root span under it.

        This is an imperative API to manually create a new span under a specific trace id and
        parent span, unlike the higher-level APIs like :py:func:`@mlflow.trace <mlflow.trace>`
        and :py:func:`with mlflow.start_span() <mlflow.start_span>`, which automatically manage
        the span lifecycle and parent-child relationship. You only need to call this method
        when using the :py:func:`start_span() <start_span>` method of MlflowClient to create
        spans.

        .. attention::

            A trace started with this method must be ended by calling
            ``MlflowClient().end_trace(trace_id)``. Otherwise the trace will be not recorded.

        Args:
            name: The name of the trace (and the root span).
            span_type: The type of the span.
            inputs: Inputs to set on the root span of the trace.
            attributes: A dictionary of attributes to set on the root span of the trace.
            tags: A dictionary of tags to set on the trace.
            experiment_id: The ID of the experiment to create the trace in. If not provided,
                MLflow will look for valid experiment in the following order: activated using
                :py:func:`mlflow.set_experiment() <mlflow.set_experiment>`,
                ``MLFLOW_EXPERIMENT_NAME`` environment variable, ``MLFLOW_EXPERIMENT_ID``
                environment variable, or the default experiment as defined by the tracking server.
            start_time_ns: The start time of the trace in nanoseconds since the UNIX epoch.

        Returns:
            An :py:class:`Span <mlflow.entities.Span>` object
            representing the root span of the trace.

        Example:

        .. code-block:: python
            :test:

            from mlflow import MlflowClient

            client = MlflowClient()

            root_span = client.start_trace("my_trace")
            trace_id = root_span.trace_id

            # Create a child span
            child_span = client.start_span(
                "child_span", trace_id=trace_id, parent_id=root_span.span_id
            )
            # Do something...
            client.end_span(trace_id=trace_id, span_id=child_span.span_id)

            client.end_trace(trace_id)
        """
        # Validate no active trace is set in the global context. If there is an active trace,
        # the span created by this method will be a child span under the active trace rather than
        # a root span of a new trace, which is not desired behavior.
        if span := mlflow.get_current_active_span():
            raise MlflowException(
                f"Another trace is already set in the global context with ID {span.trace_id}. "
                "It appears that you have already started a trace using fluent APIs like "
                "`@mlflow.trace()` or `with mlflow.start_span()`. However, it is not allowed "
                "to call MlflowClient.start_trace() under an active trace created by fluent APIs "
                "because it may lead to unexpected behavior. To resolve this issue, consider the "
                "following options:\n"
                " - To create a child span under the active trace, use "
                "`with mlflow.start_span()` or `MlflowClient.start_span()` instead.\n"
                " - To start multiple traces in parallel, avoid using fluent APIs "
                "and create all traces using `MlflowClient.start_trace()`.",
                error_code=BAD_REQUEST,
            )

        return start_span_no_context(
            name=name,
            span_type=span_type,
            inputs=inputs,
            attributes=attributes,
            tags=tags,
            experiment_id=experiment_id,
            start_time_ns=start_time_ns,
        )

    @request_id_backward_compatible
    def end_trace(
        self,
        trace_id: str,
        outputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
        status: Union[SpanStatus, str] = "OK",
        end_time_ns: Optional[int] = None,
    ):
        """
        End the trace with the given trace ID. This will end the root span of the trace and
        log the trace to the backend if configured.

        If any of children spans are not ended, they will be ended forcefully with the status
        ``TRACE_STATUS_UNSPECIFIED``. If the trace is already ended, this method will have
        no effect.

        Args:
            trace_id: The ID of the trace to end.
            outputs: Outputs to set on the trace.
            attributes: A dictionary of attributes to set on the trace. If the trace already
                has attributes, the new attributes will be merged with the existing ones.
                If the same key already exists, the new value will overwrite the old one.
            status: The status of the trace. This can be a
                :py:class:`SpanStatus <mlflow.entities.SpanStatus>` object or a string
                representing the status code defined in
                :py:class:`SpanStatusCode <mlflow.entities.SpanStatusCode>`
                e.g. ``"OK"``, ``"ERROR"``. The default status is OK.
            end_time_ns: The end time of the trace in nanoseconds since the UNIX epoch.
        """
        # NB: If the specified request ID is of no-op span, this means something went wrong in
        #     the span start logic. We should simply ignore it as the upstream should already
        #     have logged the error.
        if trace_id == NO_OP_SPAN_TRACE_ID:
            return

        trace_manager = InMemoryTraceManager.get_instance()
        root_span_id = trace_manager.get_root_span_id(trace_id)

        if root_span_id is None:
            trace = self.get_trace(trace_id=trace_id)
            if trace is None:
                raise MlflowException(
                    f"Trace with ID {trace_id} not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif trace.info.status in TraceStatus.end_statuses():
                raise MlflowException(
                    f"Trace with ID {trace_id} already finished.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        root_span = trace_manager.get_span_from_id(trace_id, root_span_id)
        if root_span:
            root_span.end(outputs, attributes, status, end_time_ns)

    @experimental
    def _log_trace(self, trace: Trace) -> str:
        """
        Log the complete Trace object to the backend store.

        # NB: Since the backend API is used directly here, customization of trace ID's
        # are not possible with this internal API. A backend-generated ID will be generated
        # directly with this invocation, instead of the one from the given trace object.

        Args:
            trace: The trace object to log.

        Returns:
            The trace ID of the logged trace.
        """
        from mlflow.tracking.fluent import _get_experiment_id

        # If the trace is created outside MLflow experiment (e.g. model serving), it does
        # not have an experiment ID, but we need to log it to the tracking server.
        experiment_id = trace.info.experiment_id or _get_experiment_id()

        # Create trace info entry in the backend
        # Note that the backend generates a new request ID for the trace. Currently there is
        # no way to insert the trace with a specific request ID given by the user.
        new_info = self._tracing_client.start_trace(
            experiment_id=experiment_id,
            timestamp_ms=trace.info.timestamp_ms,
            request_metadata={},
            tags={},
        )
        self._tracing_client.end_trace(
            request_id=new_info.trace_id,
            # Compute the end time of the original trace
            timestamp_ms=trace.info.timestamp_ms + trace.info.execution_time_ms,
            status=trace.info.status,
            request_metadata=trace.info.request_metadata,
            tags=trace.info.tags,
        )
        self._tracing_client._upload_trace_data(new_info, trace.data)
        return new_info.trace_id

    @request_id_backward_compatible
    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_id: str,
        span_type: str = SpanType.UNKNOWN,
        inputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
        start_time_ns: Optional[int] = None,
    ) -> Span:
        """
        Create a new span and start it without attaching it to the global trace context.

        This is an imperative API to manually create a new span under a specific trace id
        and parent span, unlike the higher-level APIs like
        :py:func:`@mlflow.trace <mlflow.trace>` decorator and
        :py:func:`with mlflow.start_span() <mlflow.start_span>` context manager, which
        automatically manage the span lifecycle and parent-child relationship.

        This API is useful for the case where the automatic context management is not
        sufficient, such as callback-based instrumentation where span start and end are
        not in the same call stack, or multi-threaded applications where the context is
        not propagated automatically.

        This API requires a parent span ID to be provided explicitly. If you haven't
        started any span yet, use the :py:func:`start_trace() <start_trace>` method to
        start a new trace and a root span.

        .. warning::

            The span created with this method needs to be ended explicitly by calling
            the :py:func:`end_span() <end_span>` method. Otherwise the span will be
            recorded with the incorrect end time and status ``TRACE_STATUS_UNSPECIFIED``.

        .. tip::

            Instead of creating a root span with the :py:func:`start_trace() <start_trace>`
            method, you can also use this method within the context of a parent span created
            by the fluent APIs like :py:func:`@mlflow.trace <mlflow.trace>` and
            :py:func:`with mlflow.start_span() <mlflow.start_span>`, by passing its span
            ids the parent. This flexibility allows you to use the imperative APIs in
            conjunction with the fluent APIs like below:

            .. code-block:: python
                :test:

                import mlflow
                from mlflow import MlflowClient

                client = MlflowClient()

                with mlflow.start_span("parent_span") as parent_span:
                    child_span = client.start_span(
                        name="child_span",
                        trace_id=parent_span.trace_id,
                        parent_id=parent_span.span_id,
                    )

                    # Do something...

                    client.end_span(
                        trace_id=parent_span.trace_id,
                        span_id=child_span.span_id,
                    )

            However, **the opposite does not work**. You cannot use the fluent APIs within
            the span created by this MlflowClient API. This is because the fluent APIs
            fetches the current span from the managed context, which is not set by the MLflow
            Client APIs. Once you create a span with the MLflow Client APIs, all children
            spans must be created with the MLflow Client APIs. Please be cautious when using
            this mixed approach, as it can lead to unexpected behavior if not used properly.

        Args:
            name: The name of the span.
            trace_id: The ID of the trace to attach the span to. This is synonym to
                trace_id` in OpenTelemetry.
            parent_id: The ID of the parent span. The parent span can be a span created by
                both fluent APIs like `with mlflow.start_span()`, and imperative APIs like this.
            span_type: The type of the span. Can be either a string or a
                :py:class:`SpanType <mlflow.entities.SpanType>` enum value.
            inputs: Inputs to set on the span.
            attributes: A dictionary of attributes to set on the span.
            start_time_ns: The start time of the span in nano seconds since the UNIX epoch.
                If not provided, the current time will be used.

        Returns:
            An :py:class:`mlflow.entities.Span` object representing the span.

        Example:

        .. code-block:: python
            :test:

            from mlflow import MlflowClient

            client = MlflowClient()

            span = client.start_trace("my_trace")

            x = 2

            # Create a child span
            child_span = client.start_span(
                "child_span",
                trace_id=span.trace_id,
                parent_id=span.span_id,
                inputs={"x": x},
            )

            y = x**2

            client.end_span(
                trace_id=child_span.trace_id,
                span_id=child_span.span_id,
                attributes={"factor": 2},
                outputs={"y": y},
            )

            client.end_trace(span.trace_id)
        """
        # If parent span is no-op span, the child should also be no-op too
        if trace_id == NO_OP_SPAN_TRACE_ID:
            return NoOpSpan()

        if not parent_id:
            raise MlflowException(
                "start_span() must be called with an explicit parent_id."
                "If you haven't started any span yet, use MLflowClient().start_trace() "
                "to start a new trace and root span.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not trace_id:
            raise MlflowException(
                "Trace ID must be provided to start a span.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        trace_manager = InMemoryTraceManager.get_instance()
        if not (parent_span := trace_manager.get_span_from_id(trace_id, parent_id)):
            raise MlflowException(
                f"Parent span with ID '{parent_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return start_span_no_context(
            name=name,
            span_type=span_type,
            parent_span=parent_span,
            inputs=inputs,
            attributes=attributes,
            start_time_ns=start_time_ns,
        )

    @request_id_backward_compatible
    def end_span(
        self,
        trace_id: str,
        span_id: str,
        outputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
        status: Union[SpanStatus, str] = "OK",
        end_time_ns: Optional[int] = None,
    ):
        """
        End the span with the given trace ID and span ID.

        Args:
            trace_id: The ID of the trace to end.
            span_id: The ID of the span to end.
            outputs: Outputs to set on the span.
            attributes: A dictionary of attributes to set on the span. If the span already has
                attributes, the new attributes will be merged with the existing ones. If the same
                key already exists, the new value will overwrite the old one.
            status: The status of the span. This can be a
                :py:class:`SpanStatus <mlflow.entities.SpanStatus>` object or a string
                representing the status code defined in
                :py:class:`SpanStatusCode <mlflow.entities.SpanStatusCode>`
                e.g. ``"OK"``, ``"ERROR"``. The default status is OK.
            end_time_ns: The end time of the span in nano seconds since the UNIX epoch.
                If not provided, the current time will be used.
        """
        span = InMemoryTraceManager.get_instance().get_span_from_id(trace_id, span_id)
        if span:
            span.end(
                outputs=outputs,
                attributes=attributes,
                status=status,
                end_time_ns=end_time_ns,
            )

    @request_id_backward_compatible
    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace ID.

        The trace can be an active one or the one that has already ended and recorded in the
        backend. Below is an example of setting a tag on an active trace. You can replace the
        ``trace_id`` parameter to set a tag on an already ended trace.

        .. code-block:: python
            :test:

            from mlflow import MlflowClient

            client = MlflowClient()

            root_span = client.start_trace("my_trace")
            client.set_trace_tag(root_span.trace_id, "key", "value")
            client.end_trace(root_span.trace_id)

        Args:
            trace_id: The ID of the trace to set the tag on.
            key: The string key of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
            value: The string value of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
        """
        self._tracing_client.set_trace_tag(trace_id, key, value)

    @request_id_backward_compatible
    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        """
        Delete a tag on the trace with the given trace ID.

        The trace can be an active one or the one that has already ended and recorded in the
        backend. Below is an example of deleting a tag on an active trace. You can replace the
        ``trace_id`` parameter to delete a tag on an already ended trace.

        .. code-block:: python
            :test:

            from mlflow import MlflowClient

            client = MlflowClient()

            root_span = client.start_trace("my_trace", tags={"key": "value"})
            client.delete_trace_tag(root_span.trace_id, "key")
            client.end_trace(root_span.trace_id)

        Args:
            trace_id: The ID of the trace to delete the tag from.
            key: The string key of the tag. Must be at most 250 characters long, otherwise
                it will be truncated when stored.
        """
        self._tracing_client.delete_trace_tag(trace_id, key)

    def log_assessment(
        self,
        trace_id: str,
        name: str,
        source: AssessmentSource,
        expectation: Optional[Expectation] = None,
        feedback: Optional[Feedback] = None,
        rationale: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        span_id: Optional[str] = None,
    ) -> Assessment:
        return self._tracing_client.log_assessment(
            trace_id=trace_id,
            name=name,
            source=source,
            expectation=expectation,
            feedback=feedback,
            rationale=rationale,
            metadata=metadata,
            span_id=span_id,
        )

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: Optional[str] = None,
        expectation: Optional[Expectation] = None,
        feedback: Optional[Feedback] = None,
        rationale: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Assessment:
        return self._tracing_client.update_assessment(
            trace_id=trace_id,
            assessment_id=assessment_id,
            name=name,
            expectation=expectation,
            feedback=feedback,
            rationale=rationale,
            metadata=metadata,
        )

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        return self._tracing_client.delete_assessment(trace_id, assessment_id)

    def search_experiments(
        self,
        view_type: int = ViewType.ACTIVE_ONLY,
        max_results: Optional[int] = SEARCH_MAX_RESULTS_DEFAULT,
        filter_string: Optional[str] = None,
        order_by: Optional[list[str]] = None,
        page_token=None,
    ) -> PagedList[Experiment]:
        """
        Search for experiments that match the specified search query.

        Args:
            view_type: One of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or ``ALL``
                defined in :py:class:`mlflow.entities.ViewType`.
            max_results: Maximum number of experiments desired. Certain server backend may apply
                its own limit.
            filter_string: Filter query string (e.g., ``"name = 'my_experiment'"``), defaults to
                searching for all experiments. The following identifiers, comparators, and logical
                operators are supported.

                Identifiers
                  - ``name``: Experiment name
                  - ``creation_time``: Experiment creation time
                  - ``last_update_time``: Experiment last update time
                  - ``tags.<tag_key>``: Experiment tag. If ``tag_key`` contains
                    spaces, it must be wrapped with backticks (e.g., ``"tags.`extra key`"``).

                Comparators for string attributes and tags
                  - ``=``: Equal to
                  - ``!=``: Not equal to
                  - ``LIKE``: Case-sensitive pattern match
                  - ``ILIKE``: Case-insensitive pattern match

                Comparators for numeric attributes
                  - ``=``: Equal to
                  - ``!=``: Not equal to
                  - ``<``: Less than
                  - ``<=``: Less than or equal to
                  - ``>``: Greater than
                  - ``>=``: Greater than or equal to

                Logical operators
                  - ``AND``: Combines two sub-queries and returns True if both of them are True.

            order_by: List of columns to order by. The ``order_by`` column can contain an optional
                ``DESC`` or ``ASC`` value (e.g., ``"name DESC"``). The default ordering is ``ASC``,
                so ``"name"`` is equivalent to ``"name ASC"``. If unspecified, defaults to
                ``["last_update_time DESC"]``, which lists experiments updated most recently first.
                The following fields are supported:

                - ``experiment_id``: Experiment ID
                - ``name``: Experiment name
                - ``creation_time``: Experiment creation time
                - ``last_update_time``: Experiment last update time

            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_experiments`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
            for the next page can be obtained via the ``token`` attribute of the object.

        .. code-block:: python
            :caption: Example

            import mlflow


            def assert_experiment_names_equal(experiments, expected_names):
                actual_names = [e.name for e in experiments if e.name != "Default"]
                assert actual_names == expected_names, (actual_names, expected_names)


            mlflow.set_tracking_uri("sqlite:///:memory:")
            client = mlflow.MlflowClient()

            # Create experiments
            for name, tags in [
                ("a", None),
                ("b", None),
                ("ab", {"k": "v"}),
                ("bb", {"k": "V"}),
            ]:
                client.create_experiment(name, tags=tags)

            # Search for experiments with name "a"
            experiments = client.search_experiments(filter_string="name = 'a'")
            assert_experiment_names_equal(experiments, ["a"])

            # Search for experiments with name starting with "a"
            experiments = client.search_experiments(filter_string="name LIKE 'a%'")
            assert_experiment_names_equal(experiments, ["ab", "a"])

            # Search for experiments with tag key "k" and value ending with "v" or "V"
            experiments = client.search_experiments(filter_string="tags.k ILIKE '%v'")
            assert_experiment_names_equal(experiments, ["bb", "ab"])

            # Search for experiments with name ending with "b" and tag {"k": "v"}
            experiments = client.search_experiments(filter_string="name LIKE '%b' AND tags.k = 'v'")
            assert_experiment_names_equal(experiments, ["ab"])

            # Sort experiments by name in ascending order
            experiments = client.search_experiments(order_by=["name"])
            assert_experiment_names_equal(experiments, ["a", "ab", "b", "bb"])

            # Sort experiments by ID in descending order
            experiments = client.search_experiments(order_by=["experiment_id DESC"])
            assert_experiment_names_equal(experiments, ["bb", "ab", "b", "a"])
        """
        return self._tracking_client.search_experiments(
            view_type=view_type,
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by,
            page_token=page_token,
        )

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by experiment_id from the backend store

        Args:
            experiment_id: The experiment ID returned from ``create_experiment``.

        Returns:
            :py:class:`mlflow.entities.Experiment`

        .. code-block:: python
          :caption: Example

          from mlflow import MlflowClient

          client = MlflowClient()
          exp_id = client.create_experiment("Experiment")
          experiment = client.get_experiment(exp_id)

          # Show experiment info
          print(f"Name: {experiment.name}")
          print(f"Experiment ID: {experiment.experiment_id}")
          print(f"Artifact Location: {experiment.artifact_location}")
          print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

        .. code-block:: text
          :caption: Output

          Name: Experiment
          Experiment ID: 1
          Artifact Location: file:///.../mlruns/1
          Lifecycle_stage: active
        """
        return self._tracking_client.get_experiment(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Retrieve an experiment by experiment name from the backend store

        Args:
            name: The experiment name, which is case sensitive.

        Returns:
            An instance of :py:class:`mlflow.entities.Experiment`
            if an experiment with the specified name exists, otherwise None.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Case-sensitive name
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Default")
            # Show experiment info
            print(f"Name: {experiment.name}")
            print(f"Experiment ID: {experiment.experiment_id}")
            print(f"Artifact Location: {experiment.artifact_location}")
            print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            Name: Default
            Experiment ID: 0
            Artifact Location: file:///.../mlruns/0
            Lifecycle_stage: active
        """
        return self._tracking_client.get_experiment_by_name(name)

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create an experiment.

        Args:
            name: The experiment name, which must be a unique string.
            artifact_location: The location to store run artifacts. If not provided, the server
                picks anappropriate default.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.ExperimentTag` objects, set as
                experiment tags upon experiment creation.

        Returns:
            String as an integer ID of the created experiment.

        .. code-block:: python
            :caption: Example

            from pathlib import Path
            from mlflow import MlflowClient

            # Create an experiment with a name that is unique and case sensitive.
            client = MlflowClient()
            experiment_id = client.create_experiment(
                "Social NLP Experiments",
                artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
                tags={"version": "v1", "priority": "P1"},
            )
            client.set_experiment_tag(experiment_id, "nlp.framework", "Spark NLP")

            # Fetch experiment metadata information
            experiment = client.get_experiment(experiment_id)
            print(f"Name: {experiment.name}")
            print(f"Experiment_id: {experiment.experiment_id}")
            print(f"Artifact Location: {experiment.artifact_location}")
            print(f"Tags: {experiment.tags}")
            print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            Name: Social NLP Experiments
            Experiment_id: 1
            Artifact Location: file:///.../mlruns
            Tags: {'version': 'v1', 'priority': 'P1', 'nlp.framework': 'Spark NLP'}
            Lifecycle_stage: active

        """
        return self._tracking_client.create_experiment(name, artifact_location, tags)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the backend store.

        This deletion is a soft-delete, not a permanent deletion. Experiment names can not be
        reused, unless the deleted experiment is permanently deleted by a database admin.

        Args:
            experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create an experiment with a name that is unique and case sensitive
            client = MlflowClient()
            experiment_id = client.create_experiment("New Experiment")
            client.delete_experiment(experiment_id)

            # Examine the deleted experiment details.
            experiment = client.get_experiment(experiment_id)
            print(f"Name: {experiment.name}")
            print(f"Artifact Location: {experiment.artifact_location}")
            print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            Name: New Experiment
            Artifact Location: file:///.../mlruns/1
            Lifecycle_stage: deleted

        """
        self._tracking_client.delete_experiment(experiment_id)

    def restore_experiment(self, experiment_id: str) -> None:
        """
        Restore a deleted experiment unless permanently deleted.

        Args:
            experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_experiment_info(experiment):
                print(f"Name: {experiment.name}")
                print(f"Experiment Id: {experiment.experiment_id}")
                print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


            # Create and delete an experiment
            client = MlflowClient()
            experiment_id = client.create_experiment("New Experiment")
            client.delete_experiment(experiment_id)

            # Examine the deleted experiment details.
            experiment = client.get_experiment(experiment_id)
            print_experiment_info(experiment)
            print("--")

            # Restore the experiment and fetch its info
            client.restore_experiment(experiment_id)
            experiment = client.get_experiment(experiment_id)
            print_experiment_info(experiment)

        .. code-block:: text
            :caption: Output

            Name: New Experiment
            Experiment Id: 1
            Lifecycle_stage: deleted
            --
            Name: New Experiment
            Experiment Id: 1
            Lifecycle_stage: active

        """
        self._tracking_client.restore_experiment(experiment_id)

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        """
        Update an experiment's name. The new name must be unique.

        Args:
            experiment_id: The experiment ID returned from ``create_experiment``.
            new_name: The new name for the experiment.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_experiment_info(experiment):
                print(f"Name: {experiment.name}")
                print(f"Experiment_id: {experiment.experiment_id}")
                print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


            # Create an experiment with a name that is unique and case sensitive
            client = MlflowClient()
            experiment_id = client.create_experiment("Social NLP Experiments")

            # Fetch experiment metadata information
            experiment = client.get_experiment(experiment_id)
            print_experiment_info(experiment)
            print("--")

            # Rename and fetch experiment metadata information
            client.rename_experiment(experiment_id, "Social Media NLP Experiments")
            experiment = client.get_experiment(experiment_id)
            print_experiment_info(experiment)

        .. code-block:: text
            :caption: Output

            Name: Social NLP Experiments
            Experiment_id: 1
            Lifecycle_stage: active
            --
            Name: Social Media NLP Experiments
            Experiment_id: 1
            Lifecycle_stage: active

        """
        self._tracking_client.rename_experiment(experiment_id, new_name)

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
        synchronous: Optional[bool] = None,
        dataset_name: Optional[str] = None,
        dataset_digest: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Optional[RunOperations]:
        """
        Log a metric against the run ID.

        Args:
            run_id: The run id to which the metric should be logged.
            key: Metric name. This string may only contain alphanumerics, underscores
                (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores will support keys up to length 250, but some may
                support larger keys.
            value: Metric value. Note that some special values such
                as +/- Infinity may be replaced by other values depending on the store. For
                example, the SQLAlchemy store replaces +/- Inf with max / min float values.
                All backend stores will support values up to length 5000, but some
                may support larger values.
            timestamp: Time when this metric was calculated. Defaults to the current system time.
            step: Integer training step (iteration) at which was the metric calculated.
                Defaults to 0.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully.
                If False, logs the metric asynchronously and returns a future representing the
                logging operation. If None, read from environment variable
                `MLFLOW_ENABLE_ASYNC_LOGGING`, which defaults to False if not set.
            dataset_name: The name of the dataset associated with the metric. If specified,
                ``dataset_digest`` must also be provided.
            dataset_digest: The digest of the dataset associated with the metric. If specified,
                ``dataset_name`` must also be provided.
            model_id: The ID of the model associated with the metric. If not specified, use the
                current active model ID set by :py:func:`mlflow.set_active_model`.

        Returns:
            When `synchronous=True` or None, returns None. When `synchronous=False`, returns an
            :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance that
            represents future for logging operation.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(r):
                print(f"run_id: {r.info.run_id}")
                print(f"metrics: {r.data.metrics}")
                print(f"status: {r.info.status}")


            # Create a run under the default experiment (whose id is '0').
            # Since these are low-level CRUD operations, this method will create a run.
            # To end the run, you'll have to explicitly end it.
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print_run_info(run)
            print("--")

            # Log the metric. Unlike mlflow.log_metric this method
            # does not start a run if one does not exist. It will log
            # the metric for the run id in the backend store.
            client.log_metric(run.info.run_id, "m", 1.5)
            client.set_terminated(run.info.run_id)
            run = client.get_run(run.info.run_id)
            print_run_info(run)

            # To log metric in async fashion
            client.log_metric(run.info.run_id, "m", 1.5, synchronous=False)

        .. code-block:: text
            :caption: Output

            run_id: 95e79843cb2c463187043d9065185e24
            metrics: {}
            status: RUNNING
            --
            run_id: 95e79843cb2c463187043d9065185e24
            metrics: {'m': 1.5}
            status: FINISHED
        """
        from mlflow.tracking.fluent import get_active_model_id

        synchronous = (
            synchronous if synchronous is not None else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )
        model_id = model_id or get_active_model_id()
        return self._tracking_client.log_metric(
            run_id,
            key,
            value,
            timestamp,
            step,
            synchronous=synchronous,
            dataset_name=dataset_name,
            dataset_digest=dataset_digest,
            model_id=model_id,
        )

    def log_param(
        self, run_id: str, key: str, value: Any, synchronous: Optional[bool] = None
    ) -> Any:
        """
        Log a parameter (e.g. model hyperparameter) against the run ID.

        Args:
            run_id: The run id to which the param should be logged.
            key: Parameter name. This string may only contain alphanumerics, underscores
                (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores support keys up to length 250, but some may
                support larger keys.
            value: Parameter value, but will be string-ified if not.
                All built-in backend stores support values up to length 6000, but some
                may support larger values.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully.
                If False, logs the metric asynchronously and returns a future representing the
                logging operation. If None, read from environment variable
                `MLFLOW_ENABLE_ASYNC_LOGGING`, which defaults to False if not set.

        Returns:
            When `synchronous=True` or None, returns parameter value. When `synchronous=False`,
            returns an :py:class:`mlflow.utils.async_logging.run_operations.RunOperations`
            instance that represents future for logging operation.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(r):
                print(f"run_id: {r.info.run_id}")
                print(f"params: {r.data.params}")
                print(f"status: {r.info.status}")


            # Create a run under the default experiment (whose id is '0').
            # Since these are low-level CRUD operations, this method will create a run.
            # To end the run, you'll have to explicitly end it.
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print_run_info(run)
            print("--")
            # Log the parameter. Unlike mlflow.log_param this method
            # does not start a run if one does not exist. It will log
            # the parameter in the backend store
            p_value = client.log_param(run.info.run_id, "p", 1)
            assert p_value == 1
            client.set_terminated(run.info.run_id)
            run = client.get_run(run.info.run_id)
            print_run_info(run)

        .. code-block:: text
            :caption: Output

            run_id: e649e49c7b504be48ee3ae33c0e76c93
            params: {}
            status: RUNNING
            --
            run_id: e649e49c7b504be48ee3ae33c0e76c93
            params: {'p': '1'}
            status: FINISHED
        """
        synchronous = (
            synchronous if synchronous is not None else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )
        if synchronous:
            self._tracking_client.log_param(run_id, key, value, synchronous=True)
            return value
        else:
            return self._tracking_client.log_param(run_id, key, value, synchronous=False)

    def set_experiment_tag(self, experiment_id: str, key: str, value: Any) -> None:
        """
        Set a tag on the experiment with the specified ID. Value is converted to a string.

        Args:
            experiment_id: String ID of the experiment.
            key: Name of the tag.
            value: Tag value (converted to a string).

        .. code-block:: python

            from mlflow import MlflowClient

            # Create an experiment and set its tag
            client = MlflowClient()
            experiment_id = client.create_experiment("Social Media NLP Experiments")
            client.set_experiment_tag(experiment_id, "nlp.framework", "Spark NLP")

            # Fetch experiment metadata information
            experiment = client.get_experiment(experiment_id)
            print(f"Name: {experiment.name}")
            print(f"Tags: {experiment.tags}")

        .. code-block:: text

            Name: Social Media NLP Experiments
            Tags: {'nlp.framework': 'Spark NLP'}

        """
        self._tracking_client.set_experiment_tag(experiment_id, key, value)

    def set_tag(
        self, run_id: str, key: str, value: Any, synchronous: Optional[bool] = None
    ) -> Optional[RunOperations]:
        """
        Set a tag on the run with the specified ID. Value is converted to a string.

        Args:
            run_id: String ID of the run.
            key: Tag name. This string may only contain alphanumerics, underscores (_), dashes (-),
                periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to
                length 250, but some may support larger keys.
            value: Tag value, but will be string-ified if not. All backend stores will support
                values up to length 5000, but some may support larger values.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully.
                If False, logs the metric asynchronously and returns a future representing the
                logging operation. If None, read from environment variable
                `MLFLOW_ENABLE_ASYNC_LOGGING`, which defaults to False if not set.

        Returns:
            When `synchronous=True` or None, returns None. When `synchronous=False`, returns an
            `mlflow.utils.async_logging.run_operations.RunOperations` instance that represents
            future for logging operation.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print(f"run_id: {run.info.run_id}")
                print(f"Tags: {run.data.tags}")


            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print_run_info(run)
            print("--")
            # Set a tag and fetch updated run info
            client.set_tag(run.info.run_id, "nlp.framework", "Spark NLP")
            run = client.get_run(run.info.run_id)
            print_run_info(run)

        .. code-block:: text
            :caption: Output

            run_id: 4f226eb5758145e9b28f78514b59a03b
            Tags: {}
            --
            run_id: 4f226eb5758145e9b28f78514b59a03b
            Tags: {'nlp.framework': 'Spark NLP'}
        """
        synchronous = (
            synchronous if synchronous is not None else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )
        return self._tracking_client.set_tag(run_id, key, value, synchronous=synchronous)

    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag from a run. This is irreversible.

        Args:
            run_id: String ID of the run.
            key: Name of the tag.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print(f"run_id: {run.info.run_id}")
                print(f"Tags: {run.data.tags}")


            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            tags = {"t1": 1, "t2": 2}
            experiment_id = "0"
            run = client.create_run(experiment_id, tags=tags)
            print_run_info(run)
            print("--")

            # Delete tag and fetch updated info
            client.delete_tag(run.info.run_id, "t1")
            run = client.get_run(run.info.run_id)
            print_run_info(run)

        .. code-block:: text
            :caption: Output

            run_id: b7077267a59a45d78cd9be0de4bc41f5
            Tags: {'t2': '2', 't1': '1'}
            --
            run_id: b7077267a59a45d78cd9be0de4bc41f5
            Tags: {'t2': '2'}

        """
        self._tracking_client.delete_tag(run_id, key)

    def update_run(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """Update a run with the specified ID to a new status or name.

        Args:
            run_id: The ID of the Run to update.
            status: The new status of the run to set, if specified. At least one of ``status`` or
                ``name`` should be specified.
            name: The new name of the run to set, if specified. At least one of ``name`` or
                ``status`` should be specified.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print(f"run_id: {run.info.run_id}")
                print(f"run_name: {run.info.run_name}")
                print(f"status: {run.info.status}")


            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print_run_info(run)
            print("--")

            # Update run and fetch info
            client.update_run(run.info.run_id, "FINISHED", "new_name")
            run = client.get_run(run.info.run_id)
            print_run_info(run)

        .. code-block:: text
            :caption: Output

            run_id: 1cf6bf8bf6484dd8a598bd43be367b20
            run_name: judicious-hog-915
            status: RUNNING
            --
            run_id: 1cf6bf8bf6484dd8a598bd43be367b20
            run_name: new_name
            status: FINISHED

        """
        self._tracking_client.update_run(run_id, status, name)

    def log_batch(
        self,
        run_id: str,
        metrics: Sequence[Metric] = (),
        params: Sequence[Param] = (),
        tags: Sequence[RunTag] = (),
        synchronous: Optional[bool] = None,
    ) -> Optional[RunOperations]:
        """
        Log multiple metrics, params, and/or tags.

        Args:
            run_id: String ID of the run
            metrics: If provided, List of Metric(key, value, timestamp) instances.
            params: If provided, List of Param(key, value) instances.
            tags: If provided, List of RunTag(key, value) instances.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully.
                If False, logs the metric asynchronously and returns a future representing the
                logging operation. If None, read from environment variable
                `MLFLOW_ENABLE_ASYNC_LOGGING`, which defaults to False if not set.

        Raises:
            mlflow.MlflowException: If any errors occur.

        Returns:
            When `synchronous=True` or None, returns None. When `synchronous=False`, returns an
            :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance that
            represents future for logging operation.

        .. code-block:: python
            :caption: Example

            import time

            from mlflow import MlflowClient
            from mlflow.entities import Metric, Param, RunTag


            def print_run_info(r):
                print(f"run_id: {r.info.run_id}")
                print(f"params: {r.data.params}")
                print(f"metrics: {r.data.metrics}")
                print(f"tags: {r.data.tags}")
                print(f"status: {r.info.status}")


            # Create MLflow entities and a run under the default experiment (whose id is '0').
            timestamp = int(time.time() * 1000)
            metrics = [Metric("m", 1.5, timestamp, 1)]
            params = [Param("p", "p")]
            tags = [RunTag("t", "t")]
            experiment_id = "0"
            client = MlflowClient()
            run = client.create_run(experiment_id)

            # Log entities, terminate the run, and fetch run status
            client.log_batch(run.info.run_id, metrics=metrics, params=params, tags=tags)
            client.set_terminated(run.info.run_id)
            run = client.get_run(run.info.run_id)
            print_run_info(run)

            # To log metric in async fashion
            client.log_metric(run.info.run_id, "m", 1.5, synchronous=False)

        .. code-block:: text
            :caption: Output

            run_id: ef0247fa3205410595acc0f30f620871
            params: {'p': 'p'}
            metrics: {'m': 1.5}
            tags: {'t': 't'}
            status: FINISHED

        """
        synchronous = (
            synchronous if synchronous is not None else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )

        # Stringify the values of the params
        params = [Param(key=param.key, value=str(param.value)) for param in params]

        return self._tracking_client.log_batch(
            run_id, metrics, params, tags, synchronous=synchronous
        )

    def log_inputs(
        self,
        run_id: str,
        datasets: Optional[Sequence[DatasetInput]] = None,
        models: Optional[Sequence[LoggedModelInput]] = None,
    ) -> None:
        """
        Log one or more dataset inputs to a run.

        Args:
            run_id: String ID of the run.
            datasets: List of :py:class:`mlflow.entities.DatasetInput` instances to log.
            models: List of :py:class:`mlflow.entities.LoggedModelInput` instances to log.

        Raises:
            mlflow.MlflowException: If any errors occur.
        """
        self._tracking_client.log_inputs(run_id, datasets, models)

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]):
        self._tracking_client.log_outputs(run_id, models)

    def log_artifact(self, run_id, local_path, artifact_path=None) -> None:
        """Write a local file or directory to the remote ``artifact_uri``.

        Args:
            run_id: String ID of run.
            local_path: Path to the file or directory to write.
            artifact_path: If provided, the directory in ``artifact_uri`` to write to.

        .. code-block:: python
            :caption: Example

            import tempfile
            from pathlib import Path

            from mlflow import MlflowClient

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)

            # log and fetch the artifact
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir, "features.txt")
                path.write_text(features)
                client.log_artifact(run.info.run_id, path)

            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print(f"artifact: {artifact.path}")
                print(f"is_dir: {artifact.is_dir}")
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: features.txt
            is_dir: False

        """
        if run_id.startswith(TRACE_REQUEST_ID_PREFIX):
            raise MlflowException(
                f"Invalid run id: {run_id}. `log_artifact` run id must map to a valid run."
            )
        self._tracking_client.log_artifact(run_id, local_path, artifact_path)

    def log_artifacts(
        self, run_id: str, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """Write a directory of files to the remote ``artifact_uri``.

        Args:
            run_id: String ID of run.
            local_dir: Path to the directory of files to write.
            artifact_path: If provided, the directory in ``artifact_uri`` to write to.

        .. code-block:: python
            :caption: Example

            import json
            import tempfile
            from pathlib import Path

            # Create some artifacts data to preserve
            features = "rooms, zipcode, median_price, school_rating, transport"
            data = {"state": "TX", "Available": 25, "Type": "Detached"}
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                with (tmp_dir / "data.json").open("w") as f:
                    json.dump(data, f, indent=2)
                with (tmp_dir / "features.json").open("w") as f:
                    f.write(features)

                # Create a run under the default experiment (whose id is '0'), and log
                # all files in "data" to root artifact_uri/states
                client = MlflowClient()
                experiment_id = "0"
                run = client.create_run(experiment_id)
                client.log_artifacts(run.info.run_id, tmp_dir, artifact_path="states")

            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print(f"artifact: {artifact.path}")
                print(f"is_dir: {artifact.is_dir}")
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: states
            is_dir: True

        """
        self._tracking_client.log_artifacts(run_id, local_dir, artifact_path)

    @contextlib.contextmanager
    def _log_artifact_helper(self, run_id, artifact_file):
        """Yields a temporary path to store a file, and then calls `log_artifact` against that path.

        Args:
            run_id: String ID of the run.
            artifact_file: The run-relative artifact file path in posixpath format.

        Returns:
            Temporary path to store a file.

        """
        norm_path = posixpath.normpath(artifact_file)
        filename = posixpath.basename(norm_path)
        artifact_dir = posixpath.dirname(norm_path)
        artifact_dir = None if artifact_dir == "" else artifact_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, filename)
            yield tmp_path
            self.log_artifact(run_id, tmp_path, artifact_dir)

    def _log_artifact_async_helper(self, run_id, artifact_file, artifact):
        """Log artifact asynchronously.

        Args:
            run_id: The unique identifier for the run. This ID is used to associate the
                artifact with a specific run.
            artifact_file: The file path of the artifact relative to the run's directory.
                The path should be in POSIX format, using forward slashes (/) as directory
                separators.
            artifact: The artifact to be logged.
        """
        norm_path = posixpath.normpath(artifact_file)
        filename = posixpath.basename(norm_path)
        artifact_dir = posixpath.dirname(norm_path)
        artifact_dir = None if artifact_dir == "" else artifact_dir
        self._tracking_client._log_artifact_async(run_id, filename, artifact_dir, artifact)

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        """Log text as an artifact.

        Args:
            run_id: String ID of the run.
            text: String containing text to log.
            artifact_file: The run-relative artifact file path in posixpath format to which
                the text is saved (e.g. "dir/file.txt").

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            client = MlflowClient()
            run = client.create_run(experiment_id="0")

            # Log text to a file under the run's root artifact directory
            client.log_text(run.info.run_id, "text1", "file1.txt")

            # Log text in a subdirectory of the run's root artifact directory
            client.log_text(run.info.run_id, "text2", "dir/file2.txt")

            # Log HTML text
            client.log_text(run.info.run_id, "<h1>header</h1>", "index.html")

        """
        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)

    def log_dict(self, run_id: str, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a JSON/YAML-serializable object (e.g. `dict`) as an artifact. The serialization
        format (JSON or YAML) is automatically inferred from the extension of `artifact_file`.
        If the file extension doesn't exist or match any of [".json", ".yml", ".yaml"],
        JSON format is used, and we stringify objects that can't be JSON-serialized.

        Args:
            run_id: String ID of the run.
            dictionary: Dictionary to log.
            artifact_file: The run-relative artifact file path in posixpath format to which
                the dictionary is saved (e.g. "dir/data.json").

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            client = MlflowClient()
            run = client.create_run(experiment_id="0")
            run_id = run.info.run_id

            dictionary = {"k": "v"}

            # Log a dictionary as a JSON file under the run's root artifact directory
            client.log_dict(run_id, dictionary, "data.json")

            # Log a dictionary as a YAML file in a subdirectory of the run's root artifact directory
            client.log_dict(run_id, dictionary, "dir/data.yml")

            # If the file extension doesn't exist or match any of [".json", ".yaml", ".yml"],
            # JSON format is used.
            mlflow.log_dict(run_id, dictionary, "data")
            mlflow.log_dict(run_id, dictionary, "data.txt")

        """
        extension = os.path.splitext(artifact_file)[1]

        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            with open(tmp_path, "w") as f:
                # Specify `indent` to prettify the output
                if extension in [".yml", ".yaml"]:
                    yaml.dump(dictionary, f, indent=2, default_flow_style=False)
                else:
                    # Stringify objects that can't be JSON-serialized
                    json.dump(dictionary, f, indent=2, default=str)

    def log_figure(
        self,
        run_id: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        artifact_file: str,
        *,
        save_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a figure as an artifact. The following figure objects are supported:

        - `matplotlib.figure.Figure`_
        - `plotly.graph_objects.Figure`_

        .. _matplotlib.figure.Figure:
            https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html

        .. _plotly.graph_objects.Figure:
            https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html

        Args:
            run_id: String ID of the run.
            figure: Figure to log.
            artifact_file: The run-relative artifact file path in posixpath format to which
                the figure is saved (e.g. "dir/file.png").
            save_kwargs: Additional keyword arguments passed to the method that saves the figure.

        .. code-block:: python
            :caption: Matplotlib Example

            import mlflow
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([0, 1], [2, 3])

            run = client.create_run(experiment_id="0")
            client.log_figure(run.info.run_id, fig, "figure.png")

        .. code-block:: python
            :caption: Plotly Example

            import mlflow
            from plotly import graph_objects as go

            fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

            run = client.create_run(experiment_id="0")
            client.log_figure(run.info.run_id, fig, "figure.html")

        """

        def _is_matplotlib_figure(fig):
            import matplotlib.figure

            return isinstance(fig, matplotlib.figure.Figure)

        def _is_plotly_figure(fig):
            import plotly

            return isinstance(fig, plotly.graph_objects.Figure)

        save_kwargs = save_kwargs or {}
        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            # `is_matplotlib_figure` is executed only when `matplotlib` is found in `sys.modules`.
            # This allows logging a `plotly` figure in an environment where `matplotlib` is not
            # installed.
            if "matplotlib" in sys.modules and _is_matplotlib_figure(figure):
                figure.savefig(tmp_path, **save_kwargs)
            elif "plotly" in sys.modules and _is_plotly_figure(figure):
                file_extension = os.path.splitext(artifact_file)[1]
                if file_extension == ".html":
                    save_kwargs.setdefault("include_plotlyjs", "cdn")
                    save_kwargs.setdefault("auto_open", False)
                    figure.write_html(tmp_path, **save_kwargs)
                elif file_extension in [".png", ".jpeg", ".webp", ".svg", ".pdf"]:
                    figure.write_image(tmp_path, **save_kwargs)
                else:
                    raise TypeError(
                        f"Unsupported file extension for plotly figure: '{file_extension}'"
                    )
            else:
                raise TypeError(f"Unsupported figure object type: '{type(figure)}'")

    def log_image(
        self,
        run_id: str,
        image: Union["numpy.ndarray", "PIL.Image.Image", "mlflow.Image"],
        artifact_file: Optional[str] = None,
        key: Optional[str] = None,
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
        synchronous: Optional[bool] = None,
    ) -> None:
        """
        Logs an image in MLflow, supporting two use cases:

        1. Time-stepped image logging:
            Ideal for tracking changes or progressions through iterative processes (e.g.,
            during model training phases).

            - Usage: :code:`log_image(image, key=key, step=step, timestamp=timestamp)`

        2. Artifact file image logging:
            Best suited for static image logging where the image is saved directly as a file
            artifact.

            - Usage: :code:`log_image(image, artifact_file)`

        The following image formats are supported:
            - `numpy.ndarray`_
            - `PIL.Image.Image`_

            .. _numpy.ndarray:
                https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

            .. _PIL.Image.Image:
                https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image

            - :class:`mlflow.Image`: An MLflow wrapper around PIL image for convenient image
              logging.

        Numpy array support
            - data types:

                - bool (useful for logging image masks)
                - integer [0, 255]
                - unsigned integer [0, 255]
                - float [0.0, 1.0]

                .. warning::

                    - Out-of-range integer values will raise ValueError.
                    - Out-of-range float values will auto-scale with min/max and warn.

            - shape (H: height, W: width):

                - H x W (Grayscale)
                - H x W x 1 (Grayscale)
                - H x W x 3 (an RGB channel order is assumed)
                - H x W x 4 (an RGBA channel order is assumed)

        Args:
            run_id: String ID of run.
            image: The image object to be logged.
            artifact_file: Specifies the path, in POSIX format, where the image
                will be stored as an artifact relative to the run's root directory (for
                example, "dir/image.png"). This parameter is kept for backward compatibility
                and should not be used together with `key`, `step`, or `timestamp`.
            key: Image name for time-stepped image logging. This string may only contain
                alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and
                slashes (/).
            step: Integer training step (iteration) at which the image was saved.
                Defaults to 0.
            timestamp: Time when this image was saved. Defaults to the current system time.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully.
                If False, logs the metric asynchronously and returns a future representing the
                logging operation. If None, read from environment variable
                `MLFLOW_ENABLE_ASYNC_LOGGING`, which defaults to False if not set.

        .. code-block:: python
            :caption: Time-stepped image logging numpy example

            import mlflow
            import numpy as np

            image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
            with mlflow.start_run() as run:
                client = mlflow.MlflowClient()
                client.log_image(run.info.run_id, image, key="dogs", step=3)

        .. code-block:: python
            :caption: Time-stepped image logging pillow example

            import mlflow
            from PIL import Image

            image = Image.new("RGB", (100, 100))
            with mlflow.start_run() as run:
                client = mlflow.MlflowClient()
                client.log_image(run.info.run_id, image, key="dogs", step=3)

        .. code-block:: python
            :caption: Time-stepped image logging with mlflow.Image example

            import mlflow
            from PIL import Image

            # Saving an image to retrieve later.
            Image.new("RGB", (100, 100)).save("image.png")

            image = mlflow.Image("image.png")
            with mlflow.start_run() as run:
                client = mlflow.MlflowClient()
                client.log_image(run.info.run_id, image, key="dogs", step=3)

        .. code-block:: python
            :caption: Legacy artifact file image logging numpy example

            import mlflow
            import numpy as np

            image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
            with mlflow.start_run() as run:
                client = mlflow.MlflowClient()
                client.log_image(run.info.run_id, image, "image.png")

        .. code-block:: python
            :caption: Legacy artifact file image logging pillow example

            import mlflow
            from PIL import Image

            image = Image.new("RGB", (100, 100))
            with mlflow.start_run() as run:
                client = mlflow.MlflowClient()
                client.log_image(run.info.run_id, image, "image.png")
        """
        synchronous = (
            synchronous if synchronous is not None else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )
        if artifact_file is not None and any(arg is not None for arg in [key, step, timestamp]):
            raise TypeError(
                "The `artifact_file` parameter cannot be used in conjunction with `key`, "
                "`step`, or `timestamp` parameters. Please ensure that `artifact_file` is "
                "specified alone, without any of these conflicting parameters."
            )
        elif artifact_file is None and key is None:
            raise TypeError(
                "Invalid arguments: Please specify exactly one of `artifact_file` or `key`. Use "
                "`key` to log dynamic image charts or `artifact_file` for saving static images. "
            )

        import numpy as np

        # Convert image type to PIL if its a numpy array
        if isinstance(image, np.ndarray):
            image = convert_to_pil_image(image)
        elif isinstance(image, Image):
            image = image.to_pil()
        else:
            # Import PIL and check if the image is a PIL image
            import PIL.Image

            if not isinstance(image, PIL.Image.Image):
                raise TypeError(
                    f"Unsupported image object type: {type(image)}. "
                    "`image` must be one of numpy.ndarray, "
                    "PIL.Image.Image, and mlflow.Image."
                )

        if artifact_file is not None:
            with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
                image.save(tmp_path)

        elif key is not None:
            # Check image key for invalid characters
            if not re.match(r"^[a-zA-Z0-9_\-./ ]+$", key):
                raise ValueError(
                    "The `key` parameter may only contain alphanumerics, underscores (_), "
                    "dashes (-), periods (.), spaces ( ), and slashes (/)."
                    f"The provided key `{key}` contains invalid characters."
                )

            step = step or 0
            timestamp = timestamp or get_current_time_millis()

            # Sanitize key to use in filename (replace / with # to avoid subdirectories)
            sanitized_key = re.sub(r"/", "#", key)
            filename_uuid = str(uuid.uuid4())
            # TODO: reconsider the separator used here since % has special meaning in URL encoding.
            # See https://github.com/mlflow/mlflow/issues/14136 for more details.
            # Construct a filename uuid that does not start with hex digits
            filename_uuid = f"{random.choice(string.ascii_lowercase[6:])}{filename_uuid[1:]}"
            uncompressed_filename = (
                f"images/{sanitized_key}%step%{step}%timestamp%{timestamp}%{filename_uuid}"
            )
            compressed_filename = f"{uncompressed_filename}%compressed"

            # Save full-resolution image
            image_filepath = f"{uncompressed_filename}.png"
            compressed_image_filepath = f"{compressed_filename}.webp"

            # Need to make a resize copy before running thread for thread safety
            # If further optimization is needed, we can move this resize to async queue.
            compressed_image = compress_image_size(image)

            if synchronous:
                with self._log_artifact_helper(run_id, image_filepath) as tmp_path:
                    image.save(tmp_path)
            else:
                self._log_artifact_async_helper(run_id, image_filepath, image)

            if synchronous:
                with self._log_artifact_helper(run_id, compressed_image_filepath) as tmp_path:
                    compressed_image.save(tmp_path)
            else:
                self._log_artifact_async_helper(run_id, compressed_image_filepath, compressed_image)

            # Log tag indicating that the run includes logged image
            self.set_tag(run_id, MLFLOW_LOGGED_IMAGES, True, synchronous)

    def _check_artifact_file_string(self, artifact_file: str):
        """Check if the artifact_file contains any forbidden characters.

        Args:
            artifact_file: The run-relative artifact file path in posixpath format to which
                the table is saved (e.g. "dir/file.json").
        """
        characters_to_check = ['"', "'", ",", ":", "[", "]", "{", "}"]
        for char in characters_to_check:
            if char in artifact_file:
                raise ValueError(f"The artifact_file contains forbidden character: {char}")

    def _read_from_file(self, artifact_path):
        import pandas as pd

        if artifact_path.endswith(".json"):
            return pd.read_json(artifact_path, orient="split")
        if artifact_path.endswith(".parquet"):
            return pd.read_parquet(artifact_path)
        raise ValueError(f"Unsupported file type in {artifact_path}. Expected .json or .parquet")

    def log_table(
        self,
        run_id: str,
        data: Union[dict[str, Any], "pandas.DataFrame"],
        artifact_file: str,
    ) -> None:
        """
        Log a table to MLflow Tracking as a JSON artifact. If the artifact_file already exists
        in the run, the data would be appended to the existing artifact_file.

        Args:
            run_id: String ID of the run.
            data: Dictionary or pandas.DataFrame to log.
            artifact_file: The run-relative artifact file path in posixpath format to which
                the table is saved (e.g. "dir/file.json").

        .. code-block:: python
            :test:
            :caption: Dictionary Example

            import mlflow
            from mlflow import MlflowClient

            table_dict = {
                "inputs": ["What is MLflow?", "What is Databricks?"],
                "outputs": ["MLflow is ...", "Databricks is ..."],
                "toxicity": [0.0, 0.0],
            }
            with mlflow.start_run() as run:
                client = MlflowClient()
                client.log_table(
                    run.info.run_id, data=table_dict, artifact_file="qabot_eval_results.json"
                )

        .. code-block:: python
            :test:
            :caption: Pandas DF Example

            import mlflow
            import pandas as pd
            from mlflow import MlflowClient

            table_dict = {
                "inputs": ["What is MLflow?", "What is Databricks?"],
                "outputs": ["MLflow is ...", "Databricks is ..."],
                "toxicity": [0.0, 0.0],
            }
            df = pd.DataFrame.from_dict(table_dict)
            with mlflow.start_run() as run:
                client = MlflowClient()
                client.log_table(run.info.run_id, data=df, artifact_file="qabot_eval_results.json")

        .. code-block:: python
            :test:
            :caption: Image Column Example

            import mlflow
            import pandas as pd
            from mlflow import MlflowClient

            image = mlflow.Image([[1, 2, 3]])
            table_dict = {
                "inputs": ["Show me a dog", "Show me a cat"],
                "outputs": [image, image],
            }
            df = pd.DataFrame.from_dict(table_dict)
            with mlflow.start_run() as run:
                client = MlflowClient()
                client.log_table(run.info.run_id, data=df, artifact_file="image_gen.json")
        """
        import pandas as pd

        self._check_artifact_file_string(artifact_file)
        if not artifact_file.endswith((".json", ".parquet")):
            raise ValueError(
                f"Invalid artifact file path '{artifact_file}'. Please ensure the file you are "
                "trying to log as a table has a file name with either '.json' "
                "or '.parquet' extension."
            )

        if not isinstance(data, (pd.DataFrame, dict)):
            raise MlflowException.invalid_parameter_value(
                "data must be a pandas.DataFrame or a dictionary"
            )

        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            # catch error `If using all scalar values, you must pass an index`
            # for data like {"inputs": "What is MLflow?"}
            except ValueError:
                data = pd.DataFrame([data])

        # Check if the column is a `PIL.Image.Image` or `mlflow.Image` object
        # and save filepath
        if len(data.select_dtypes(include=["object"]).columns) > 0:

            def process_image(image):
                # remove extension from artifact_file
                table_name, _ = os.path.splitext(artifact_file)
                # save image to path
                filepath = posixpath.join("table_images", table_name, str(uuid.uuid4()))
                image_filepath = filepath + ".png"
                compressed_image_filepath = filepath + ".webp"
                with self._log_artifact_helper(run_id, image_filepath) as artifact_path:
                    image.save(artifact_path)

                # save compressed image to path
                compressed_image = compress_image_size(image)

                with self._log_artifact_helper(run_id, compressed_image_filepath) as artifact_path:
                    compressed_image.save(artifact_path)

                # return a dictionary object indicating its an image path
                return {
                    "type": "image",
                    "filepath": image_filepath,
                    "compressed_filepath": compressed_image_filepath,
                }

            def check_is_image_object(obj):
                return (
                    hasattr(obj, "save")
                    and callable(getattr(obj, "save"))
                    and hasattr(obj, "resize")
                    and callable(getattr(obj, "resize"))
                    and hasattr(obj, "size")
                )

            for column in data.columns:
                isImage = data[column].map(lambda x: check_is_image_object(x))
                if any(isImage) and not all(isImage):
                    raise ValueError(
                        f"Column `{column}` contains a mix of images and non-images. "
                        "Please ensure that all elements in the column are of the same type."
                    )
                elif all(isImage):
                    # Save files to artifact storage
                    data[column] = data[column].map(lambda x: process_image(x))

        def write_to_file(data, artifact_path):
            if artifact_path.endswith(".json"):
                data.to_json(artifact_path, orient="split", index=False, date_format="iso")
            elif artifact_path.endswith(".parquet"):
                data.to_parquet(artifact_path, index=False)

        norm_path = posixpath.normpath(artifact_file)
        artifact_dir = posixpath.dirname(norm_path)
        artifact_dir = None if artifact_dir == "" else artifact_dir
        artifacts = [f.path for f in self.list_artifacts(run_id, path=artifact_dir)]
        if artifact_file in artifacts:
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=artifact_file, dst_path=tmpdir
                )
                existing_predictions = self._read_from_file(downloaded_artifact_path)
            data = pd.concat([existing_predictions, data], ignore_index=True)
            _logger.debug(
                "Appending new table to already existing artifact "
                f"{artifact_file} for run {run_id}."
            )

        with self._log_artifact_helper(run_id, artifact_file) as artifact_path:
            try:
                write_to_file(data, artifact_path)
            except Exception as e:
                raise MlflowException(
                    f"Failed to save {data} as table as the data is not JSON serializable. "
                    f"Error: {e}"
                )

        run = self.get_run(run_id)

        # Get the current value of the tag
        current_tag_value = json.loads(run.data.tags.get(MLFLOW_LOGGED_ARTIFACTS, "[]"))
        tag_value = {"path": artifact_file, "type": "table"}

        # Append the new tag value to the list if one doesn't exists
        if tag_value not in current_tag_value:
            current_tag_value.append(tag_value)
            # Set the tag with the updated list
            self.set_tag(run_id, MLFLOW_LOGGED_ARTIFACTS, json.dumps(current_tag_value))

    def load_table(
        self,
        experiment_id: str,
        artifact_file: str,
        run_ids: Optional[list[str]] = None,
        extra_columns: Optional[list[str]] = None,
    ) -> "pandas.DataFrame":
        """
        Load a table from MLflow Tracking as a pandas.DataFrame. The table is loaded from the
        specified artifact_file in the specified run_ids. The extra_columns are columns that
        are not in the table but are augmented with run information and added to the DataFrame.

        Args:
            experiment_id: The experiment ID to load the table from.
            artifact_file: The run-relative artifact file path in posixpath format to which
                table to load (e.g. "dir/file.json").
            run_ids: Optional list of run_ids to load the table from. If no run_ids are
                specified, the table is loaded from all runs in the current experiment.
            extra_columns: Optional list of extra columns to add to the returned DataFrame
                For example, if extra_columns=["run_id"], then the returned DataFrame
                will have a column named run_id.

        Returns:
            pandas.DataFrame containing the loaded table if the artifact exists
            or else throw a MlflowException.

         .. code-block:: python
            :test:
            :caption: Example with passing run_ids

            import mlflow
            import pandas as pd
            from mlflow import MlflowClient

            table_dict = {
                "inputs": ["What is MLflow?", "What is Databricks?"],
                "outputs": ["MLflow is ...", "Databricks is ..."],
                "toxicity": [0.0, 0.0],
            }
            df = pd.DataFrame.from_dict(table_dict)
            client = MlflowClient()
            run = client.create_run(experiment_id="0")
            client.log_table(run.info.run_id, data=df, artifact_file="qabot_eval_results.json")
            loaded_table = client.load_table(
                experiment_id="0",
                artifact_file="qabot_eval_results.json",
                run_ids=[
                    run.info.run_id,
                ],
                # Append a column containing the associated run ID for each row
                extra_columns=["run_id"],
            )

        .. code-block:: python
            :test:
            :caption: Example with passing no run_ids

            # Loads the table with the specified name for all runs in the given
            # experiment and joins them together
            import mlflow
            import pandas as pd
            from mlflow import MlflowClient

            table_dict = {
                "inputs": ["What is MLflow?", "What is Databricks?"],
                "outputs": ["MLflow is ...", "Databricks is ..."],
                "toxicity": [0.0, 0.0],
            }
            df = pd.DataFrame.from_dict(table_dict)
            client = MlflowClient()
            run = client.create_run(experiment_id="0")
            client.log_table(run.info.run_id, data=df, artifact_file="qabot_eval_results.json")
            loaded_table = client.load_table(
                experiment_id="0",
                artifact_file="qabot_eval_results.json",
                # Append the run ID and the parent run ID to the table
                extra_columns=["run_id"],
            )
        """
        import pandas as pd

        self._check_artifact_file_string(artifact_file)
        subset_tag_value = f'"path"%:%"{artifact_file}",%"type"%:%"table"'

        # Build the filter string
        filter_string = f"tags.{MLFLOW_LOGGED_ARTIFACTS} LIKE '%{subset_tag_value}%'"
        if run_ids:
            list_run_ids = ",".join(map(repr, run_ids))
            filter_string += f" and attributes.run_id IN ({list_run_ids})"

        runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)
        if run_ids and len(run_ids) != len(runs):
            _logger.warning(
                "Not all runs have the specified table artifact. Some runs will be skipped."
            )

        # TODO: Add parallelism support here
        def get_artifact_data(run):
            run_id = run.run_id
            norm_path = posixpath.normpath(artifact_file)
            artifact_dir = posixpath.dirname(norm_path)
            artifact_dir = None if artifact_dir == "" else artifact_dir
            existing_predictions = pd.DataFrame()

            artifacts = [
                f.path for f in self.list_artifacts(run_id, path=artifact_dir) if not f.is_dir
            ]
            if artifact_file in artifacts:
                with tempfile.TemporaryDirectory() as tmpdir:
                    downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_file,
                        dst_path=tmpdir,
                    )
                    existing_predictions = self._read_from_file(downloaded_artifact_path)
                    if extra_columns is not None:
                        for column in extra_columns:
                            if column in existing_predictions:
                                column_name = f"{column}_"
                                _logger.warning(
                                    f"Column name {column} already exists in the table. "
                                    "Resolving the conflict, by appending an underscore "
                                    "to the column name."
                                )
                            else:
                                column_name = column
                            existing_predictions[column_name] = run[column]

            else:
                raise MlflowException(
                    f"Artifact {artifact_file} not found for run {run_id}.", RESOURCE_DOES_NOT_EXIST
                )

            return existing_predictions

        if not runs.empty:
            return pd.concat(
                [get_artifact_data(run) for _, run in runs.iterrows()], ignore_index=True
            )
        else:
            raise MlflowException(
                "No runs found with the corresponding table artifact.", RESOURCE_DOES_NOT_EXIST
            )

    def _record_logged_model(self, run_id, mlflow_model):
        """Record logged model info with the tracking server.

        Args:
            run_id: run_id under which the model has been logged.
            mlflow_model: Model info to be recorded.
        """
        self._tracking_client._record_logged_model(run_id, mlflow_model)

    def list_artifacts(self, run_id: str, path=None) -> list[FileInfo]:
        """List the artifacts for a run.

        Args:
            run_id: The run to list artifacts from.
            path: The run's relative artifact path to list from. By default it is set to None
                or the root artifact path.

        Returns:
            List of :py:class:`mlflow.entities.FileInfo`

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_artifact_info(artifact):
                print(f"artifact: {artifact.path}")
                print(f"is_dir: {artifact.is_dir}")
                print(f"size: {artifact.file_size}")


            features = "rooms zipcode, median_price, school_rating, transport"
            labels = "price"

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)

            # Create some artifacts and log under the above run
            for file, content in [("features", features), ("labels", labels)]:
                with open(f"{file}.txt", "w") as f:
                    f.write(content)
                client.log_artifact(run.info.run_id, f"{file}.txt")

            # Fetch the logged artifacts
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print_artifact_info(artifact)
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: features.txt
            is_dir: False
            size: 53
            artifact: labels.txt
            is_dir: False
            size: 5

        """
        return self._tracking_client.list_artifacts(run_id, path)

    def list_logged_model_artifacts(
        self, model_id: str, path: Optional[str] = None
    ) -> list[FileInfo]:
        return self._tracking_client.list_logged_model_artifacts(model_id, path)

    def download_artifacts(self, run_id: str, path: str, dst_path: Optional[str] = None) -> str:
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        Args:
            run_id: The run to download artifacts from.
            path: Relative source path to the desired artifact.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.

        Returns:
            Local path of desired artifact.

        .. code-block:: python
            :caption: Example

            import os
            import mlflow
            from mlflow import MlflowClient

            features = "rooms, zipcode, median_price, school_rating, transport"
            with open("features.txt", "w") as f:
                f.write(features)

            # Log artifacts
            with mlflow.start_run() as run:
                mlflow.log_artifact("features.txt", artifact_path="features")

            # Download artifacts
            client = MlflowClient()
            local_dir = "/tmp/artifact_downloads"
            if not os.path.exists(local_dir):
                os.mkdir(local_dir)
            local_path = client.download_artifacts(run.info.run_id, "features", local_dir)
            print(f"Artifacts downloaded in: {local_path}")
            print(f"Artifacts: {os.listdir(local_path)}")

        .. code-block:: text
            :caption: Output

            Artifacts downloaded in: /tmp/artifact_downloads/features
            Artifacts: ['features.txt']
        """
        return self._tracking_client.download_artifacts(run_id, path, dst_path)

    def set_terminated(
        self, run_id: str, status: Optional[str] = None, end_time: Optional[int] = None
    ) -> None:
        """Set a run's status to terminated.

        Args:
            run_id: The ID of the run to terminate.
            status: A string value of :py:class:`mlflow.entities.RunStatus`. Defaults to "FINISHED".
            end_time: If not provided, defaults to the current time.

        .. code-block:: python

            from mlflow import MlflowClient


            def print_run_info(r):
                print(f"run_id: {r.info.run_id}")
                print(f"status: {r.info.status}")


            # Create a run under the default experiment (whose id is '0').
            # Since this is low-level CRUD operation, this method will create a run.
            # To end the run, you'll have to explicitly terminate it.
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print_run_info(run)
            print("--")

            # Terminate the run and fetch updated status. By default,
            # the status is set to "FINISHED". Other values you can
            # set are "KILLED", "FAILED", "RUNNING", or "SCHEDULED".
            client.set_terminated(run.info.run_id, status="KILLED")
            run = client.get_run(run.info.run_id)
            print_run_info(run)

        .. code-block:: text

            run_id: 575fb62af83f469e84806aee24945973
            status: RUNNING
            --
            run_id: 575fb62af83f469e84806aee24945973
            status: KILLED

        """
        self._tracking_client.set_terminated(run_id, status, end_time)

    def delete_run(self, run_id: str) -> None:
        """Deletes a run with the given ID.

        Args:
            run_id: The unique run id to delete.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            print(f"run_id: {run_id}; lifecycle_stage: {run.info.lifecycle_stage}")
            print("--")
            client.delete_run(run_id)
            del_run = client.get_run(run_id)
            print(f"run_id: {run_id}; lifecycle_stage: {del_run.info.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            run_id: a61c7a1851324f7094e8d5014c58c8c8; lifecycle_stage: active
            run_id: a61c7a1851324f7094e8d5014c58c8c8; lifecycle_stage: deleted

        """
        self._tracking_client.delete_run(run_id)

    def restore_run(self, run_id: str) -> None:
        """Restores a deleted run with the given ID.

        Args:
            run_id: The unique run id to restore.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            print(f"run_id: {run_id}; lifecycle_stage: {run.info.lifecycle_stage}")
            client.delete_run(run_id)
            del_run = client.get_run(run_id)
            print(f"run_id: {run_id}; lifecycle_stage: {del_run.info.lifecycle_stage}")
            client.restore_run(run_id)
            rest_run = client.get_run(run_id)
            print(f"run_id: {run_id}; lifecycle_stage: {rest_run.info.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: active
            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: deleted
            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: active

        """
        self._tracking_client.restore_run(run_id)

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        run_view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[Run]:
        """
        Search for Runs that fit the specified criteria.

        Args:
            experiment_ids: List of experiment IDs, or a single int or string id.
            filter_string: Filter query string, defaults to searching all runs.
            run_view_type: one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL runs
                defined in :py:class:`mlflow.entities.ViewType`.
            max_results: Maximum number of runs desired.
            order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
                can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
                The default ordering is to sort by ``start_time DESC``, then ``run_id``.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_runs`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
            If the underlying tracking store supports pagination, the token for the next page may
            be obtained via the ``token`` attribute of the returned object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
            from mlflow.entities import ViewType


            def print_run_info(runs):
                for r in runs:
                    print(f"run_id: {r.info.run_id}")
                    print(f"lifecycle_stage: {r.info.lifecycle_stage}")
                    print(f"metrics: {r.data.metrics}")
                    # Exclude mlflow system tags
                    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
                    print(f"tags: {tags}")


            # Create an experiment and log two runs with metrics and tags under the experiment
            experiment_id = mlflow.create_experiment("Social NLP Experiments")
            with mlflow.start_run(experiment_id=experiment_id) as run:
                mlflow.log_metric("m", 1.55)
                mlflow.set_tag("s.release", "1.1.0-RC")
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_metric("m", 2.50)
                mlflow.set_tag("s.release", "1.2.0-GA")
            # Search all runs under experiment id and order them by
            # descending value of the metric 'm'
            client = MlflowClient()
            runs = client.search_runs(experiment_id, order_by=["metrics.m DESC"])
            print_run_info(runs)
            print("--")
            # Delete the first run
            client.delete_run(run_id=run.info.run_id)
            # Search only deleted runs under the experiment id and use a case insensitive pattern
            # in the filter_string for the tag.
            filter_string = "tags.s.release ILIKE '%rc%'"
            runs = client.search_runs(
                experiment_id, run_view_type=ViewType.DELETED_ONLY, filter_string=filter_string
            )
            print_run_info(runs)

        .. code-block:: text
            :caption: Output

            run_id: 0efb2a68833d4ee7860a964fad31cb3f
            lifecycle_stage: active
            metrics: {'m': 2.5}
            tags: {'s.release': '1.2.0-GA'}
            run_id: 7ab027fd72ee4527a5ec5eafebb923b8
            lifecycle_stage: active
            metrics: {'m': 1.55}
            tags: {'s.release': '1.1.0-RC'}
            --
            run_id: 7ab027fd72ee4527a5ec5eafebb923b8
            lifecycle_stage: deleted
            metrics: {'m': 1.55}
            tags: {'s.release': '1.1.0-RC'}
        """
        return self._tracking_client.search_runs(
            experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
        )

    # Registry API

    # Registered Model Methods

    def create_registered_model(
        self,
        name: str,
        tags: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
        deployment_job_id: Optional[str] = None,
    ) -> RegisteredModel:
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
            description: Description of the model.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created by backend.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_model_info(rm):
                print(f"name: {rm.name}")
                print(f"tags: {rm.tags}")
                print(f"description: {rm.description}")


            name = "SocialMediaTextAnalyzer"
            tags = {"nlp.framework": "Spark NLP"}
            desc = "This sentiment analysis model classifies the tone-happy, sad, angry."

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()
            client.create_registered_model(name, tags, desc)
            print_registered_model_info(client.get_registered_model(name))

        .. code-block:: text
            :caption: Output

            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies the tone-happy, sad, angry.

        """
        if has_prompt_tag(tags):
            raise MlflowException.invalid_parameter_value("Prompts cannot be registered as models.")

        return self._get_registry_client().create_registered_model(
            name, tags, description, deployment_job_id
        )

    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """Update registered model name.

        Args:
            name: Name of the registered model to update.
            new_name: New proposed name for the registered model.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_model_info(rm):
                print(f"name: {rm.name}")
                print(f"tags: {rm.tags}")
                print(f"description: {rm.description}")


            name = "SocialTextAnalyzer"
            tags = {"nlp.framework": "Spark NLP"}
            desc = "This sentiment analysis model classifies the tone-happy, sad, angry."

            # create a new registered model name
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()
            client.create_registered_model(name, tags, desc)
            print_registered_model_info(client.get_registered_model(name))
            print("--")

            # rename the model
            new_name = "SocialMediaTextAnalyzer"
            client.rename_registered_model(name, new_name)
            print_registered_model_info(client.get_registered_model(new_name))

        .. code-block:: text
            :caption: Output

            name: SocialTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies the tone-happy, sad, angry.
            --
            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies the tone-happy, sad, angry.
        """
        self._raise_if_prompt(name)
        self._get_registry_client().rename_registered_model(name, new_name)

    def update_registered_model(
        self, name: str, description: Optional[str] = None, deployment_job_id: Optional[str] = None
    ) -> RegisteredModel:
        """
        Updates metadata for RegisteredModel entity. Input field ``description`` should be non-None.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to update.
            description: (Optional) New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            def print_registered_model_info(rm):
                print(f"name: {rm.name}")
                print(f"tags: {rm.tags}")
                print(f"description: {rm.description}")


            name = "SocialMediaTextAnalyzer"
            tags = {"nlp.framework": "Spark NLP"}
            desc = "This sentiment analysis model classifies the tone-happy, sad, angry."

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()
            client.create_registered_model(name, tags, desc)
            print_registered_model_info(client.get_registered_model(name))
            print("--")

            # Update the model's description
            desc = "This sentiment analysis model classifies tweets' tone: happy, sad, angry."
            client.update_registered_model(name, desc)
            print_registered_model_info(client.get_registered_model(name))

        .. code-block:: text
            :caption: Output

            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies the tone-happy, sad, angry.
            --
            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies tweets' tone: happy, sad, angry.
        """
        self._raise_if_prompt(name)
        return self._get_registry_client().update_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )

    def delete_registered_model(self, name: str):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to delete.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_models_info(r_models):
                print("--")
                for rm in r_models:
                    print(f"name: {rm.name}")
                    print(f"tags: {rm.tags}")
                    print(f"description: {rm.description}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()

            # Register a couple of models with respective names, tags, and descriptions
            for name, tags, desc in [
                ("name1", {"t1": "t1"}, "description1"),
                ("name2", {"t2": "t2"}, "description2"),
            ]:
                client.create_registered_model(name, tags, desc)

            # Fetch all registered models
            print_registered_models_info(client.search_registered_models())

            # Delete one registered model and fetch again
            client.delete_registered_model("name1")
            print_registered_models_info(client.search_registered_models())

        .. code-block:: text
            :caption: Output

            --
            name: name1
            tags: {'t1': 't1'}
            description: description1
            name: name2
            tags: {'t2': 't2'}
            description: description2
            --
            name: name2
            tags: {'t2': 't2'}
            description: description2
        """
        self._raise_if_prompt(name)
        self._get_registry_client().delete_registered_model(name)

    def search_registered_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[RegisteredModel]:
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string (e.g., "name = 'a_model_name' and tag.key =
                'value1'"), defaults to searching for all registered models. The following
                identifiers, comparators, and logical operators are supported.

                Identifiers
                  - ``name``: registered model name.
                  - ``tags.<tag_key>``: registered model tag. If ``tag_key`` contains spaces, it
                    must be wrapped with backticks (e.g., "tags.`extra key`").

                Comparators
                  - ``=``: Equal to.
                  - ``!=``: Not equal to.
                  - ``LIKE``: Case-sensitive pattern match.
                  - ``ILIKE``: Case-insensitive pattern match.

                Logical operators
                  - ``AND``: Combines two sub-queries and returns True if both of them are True.

            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient()

            # Get search results filtered by the registered model name
            model_name = "CordobaWeatherForecastModel"
            filter_string = f"name='{model_name}'"
            results = client.search_registered_models(filter_string=filter_string)
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

            # Get search results filtered by the registered model name that matches
            # prefix pattern
            filter_string = "name LIKE 'Boston%'"
            results = client.search_registered_models(filter_string=filter_string)
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

            # Get all registered models and order them by ascending order of the names
            results = client.search_registered_models(order_by=["name ASC"])
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

        .. code-block:: text
            :caption: Output

            ------------------------------------------------------------------------------------
            name=CordobaWeatherForecastModel; run_id=eaef868ee3d14d10b4299c4c81ba8814; version=1
            name=CordobaWeatherForecastModel; run_id=e14afa2f47a040728060c1699968fd43; version=2
            ------------------------------------------------------------------------------------
            name=BostonWeatherForecastModel; run_id=ddc51b9407a54b2bb795c8d680e63ff6; version=1
            name=BostonWeatherForecastModel; run_id=48ac94350fba40639a993e1b3d4c185d; version=2
            -----------------------------------------------------------------------------------
            name=AzureWeatherForecastModel; run_id=5fcec6c4f1c947fc9295fef3fa21e52d; version=1
            name=AzureWeatherForecastModel; run_id=8198cb997692417abcdeb62e99052260; version=3
            name=BostonWeatherForecastModel; run_id=ddc51b9407a54b2bb795c8d680e63ff6; version=1
            name=BostonWeatherForecastModel; run_id=48ac94350fba40639a993e1b3d4c185d; version=2
            name=CordobaWeatherForecastModel; run_id=eaef868ee3d14d10b4299c4c81ba8814; version=1
            name=CordobaWeatherForecastModel; run_id=e14afa2f47a040728060c1699968fd43; version=2

        """
        return self._get_registry_client().search_registered_models(
            filter_string, max_results, order_by, page_token
        )

    def get_registered_model(self, name: str) -> RegisteredModel:
        """Get a registered model.

        Args:
            name: Name of the registered model to get.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_model_info(rm):
                print("--")
                print(f"name: {rm.name}")
                print(f"tags: {rm.tags}")
                print(f"description: {rm.description}")


            name = "SocialMediaTextAnalyzer"
            tags = {"nlp.framework": "Spark NLP"}
            desc = "This sentiment analysis model classifies the tone-happy, sad, angry."
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()
            # Create and fetch the registered model
            client.create_registered_model(name, tags, desc)
            model = client.get_registered_model(name)
            print_model_info(model)

        .. code-block:: text
            :caption: Output

            --
            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework': 'Spark NLP'}
            description: This sentiment analysis model classifies the tone-happy, sad, angry.
        """
        rm = self._get_registry_client().get_registered_model(name)

        # Prompt should not be returned as a registered model
        if has_prompt_tag(rm._tags):
            raise _model_not_found(name)

        return rm

    @deprecated(since="2.9.0", impact=_STAGES_DEPRECATION_WARNING)
    def get_latest_versions(
        self, name: str, stages: Optional[list[str]] = None
    ) -> list[ModelVersion]:
        """
        Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        Args:
            name: Name of the registered model from which to get the latest versions.
            stages: List of desired stages. If input list is None, return latest versions for
                for ALL_STAGES.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_models_info(mv):
                for m in mv:
                    print(f"name: {m.name}")
                    print(f"latest version: {m.version}")
                    print(f"run_id: {m.run_id}")
                    print(f"current_stage: {m.current_stage}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            # Create two runs Log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)
            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)
            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)
            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = f"runs:/{run_id}/sklearn-model"
                mv = client.create_model_version(name, model_uri, run_id)
                print(f"model version {mv.version} created")
            # Fetch latest version; this will be version 2
            print("--")
            print_models_info(client.get_latest_versions(name, stages=["None"]))

        .. code-block:: text
            :caption: Output

            model version 1 created
            model version 2 created
            --
            name: RandomForestRegression
            latest version: 2
            run_id: 31165664be034dc698c52a4bdeb71663
            current_stage: None
        """
        self._raise_if_prompt(name)
        return self._get_registry_client().get_latest_versions(name, stages)

    def set_registered_model_tag(self, name, key, value) -> None:
        """Set a tag for the registered model.

        Args:
            name: Registered model name.
            key: Tag key to log.
            value: Tag value log.

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_model_info(rm):
                print("--")
                print("name: {}".format(rm.name))
                print("tags: {}".format(rm.tags))


            name = "SocialMediaTextAnalyzer"
            tags = {"nlp.framework1": "Spark NLP"}
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()

            # Create registered model, set an additional tag, and fetch
            # update model info
            client.create_registered_model(name, tags, desc)
            model = client.get_registered_model(name)
            print_model_info(model)
            client.set_registered_model_tag(name, "nlp.framework2", "VADER")
            model = client.get_registered_model(name)
            print_model_info(model)

        .. code-block:: text
            :caption: Output

            --
            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework1': 'Spark NLP'}
            --
            name: SocialMediaTextAnalyzer
            tags: {'nlp.framework1': 'Spark NLP', 'nlp.framework2': 'VADER'}
        """
        self._raise_if_prompt(name)
        self._get_registry_client().set_registered_model_tag(name, key, value)

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_models_info(r_models):
                print("--")
                for rm in r_models:
                    print(f"name: {rm.name}")
                    print(f"tags: {rm.tags}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            client = MlflowClient()

            # Register a couple of models with respective names and tags
            for name, tags in [("name1", {"t1": "t1"}), ("name2", {"t2": "t2"})]:
                client.create_registered_model(name, tags)

            # Fetch all registered models
            print_registered_models_info(client.search_registered_models())
            # Delete a tag from model `name2`
            client.delete_registered_model_tag("name2", "t2")
            print_registered_models_info(client.search_registered_models())

        .. code-block:: text
            :caption: Output

            --
            name: name1
            tags: {'t1': 't1'}
            name: name2
            tags: {'t2': 't2'}
            --
            name: name1
            tags: {'t1': 't1'}
            name: name2
            tags: {}
        """
        self._raise_if_prompt(name)
        self._get_registry_client().delete_registered_model_tag(name, key)

    # Model Version Methods

    def _create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        local_model_path: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ModelVersion:
        if has_prompt_tag(tags):
            raise MlflowException.invalid_parameter_value("Prompts cannot be registered as models.")

        tracking_uri = self._tracking_client.tracking_uri
        if (
            not run_link
            and is_databricks_uri(tracking_uri)
            and tracking_uri != self._registry_uri
            and not is_databricks_unity_catalog_uri(self._registry_uri)
        ):
            if not run_id:
                eprint(
                    "Warning: no run_link will be recorded with the model version "
                    "because no run_id was given"
                )
            else:
                run_link = get_databricks_run_url(tracking_uri, run_id)
        new_source = source
        if is_databricks_uri(self._registry_uri):
            if tracking_uri != self._registry_uri:
                # Print out some info for user since the copy may take a while for large models.
                eprint(
                    "=== Copying model files from the source location to the model"
                    + " registry workspace ==="
                )
                new_source = _upload_artifacts_to_databricks(
                    source, run_id, tracking_uri, self._registry_uri
                )
                # NOTE: we can't easily delete the target temp location due to the async nature
                # of the model version creation - printing to let the user know.
                eprint(
                    f"=== Source model files were copied to {new_source}"
                    + " in the model registry workspace. You may want to delete the files once the"
                    + " model version is in 'READY' status. You can also find this location in the"
                    + " `source` field of the created model version. ==="
                )
            elif model_id is not None:
                logged_model = self.get_logged_model(model_id)
                # models:/<model_id> source is not supported by WSMR
                new_source = logged_model.artifact_location

        return self._get_registry_client().create_model_version(
            name=name,
            source=new_source,
            run_id=run_id,
            tags=tags,
            run_link=run_link,
            description=description,
            await_creation_for=await_creation_for,
            local_model_path=local_model_path,
            model_id=model_id,
        )

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        model_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create a new model version from given source.

        Args:
            name: Name for the containing registered model.
            source: URI indicating the location of the model artifacts. The artifact URI can be
                run relative (e.g. ``runs:/<run_id>/<model_artifact_path>``), a model
                registry URI (e.g. ``models:/<model_name>/<version>``), or other URIs
                supported by the model registry backend (e.g. `"s3://my_bucket/my/model"`).
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            await_creation_for: Number of seconds to wait for the model version to finish being
                created and is in ``READY`` status. By default, the function
                waits for five minutes. Specify 0 or None to skip waiting.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                      registered model version, if applicable.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
            backend.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            desc = "A new version of the model"
            runs_uri = f"runs:/{run.info.run_id}/sklearn-model"
            model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
            mv = client.create_model_version(name, model_src, run.info.run_id, description=desc)
            print(f"Name: {mv.name}")
            print(f"Version: {mv.version}")
            print(f"Description: {mv.description}")
            print(f"Status: {mv.status}")
            print(f"Stage: {mv.current_stage}")

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Description: A new version of the model
            Status: READY
            Stage: None
        """
        return self._create_model_version(
            name=name,
            source=source,
            run_id=run_id,
            tags=tags,
            run_link=run_link,
            description=description,
            await_creation_for=await_creation_for,
            model_id=model_id,
        )

    def copy_model_version(self, src_model_uri, dst_name) -> ModelVersion:
        """
        Copy a model version from one registered model to another as a new model version.

        Args:
            src_model_uri: The model URI of the model version to copy. This must be a model
                registry URI with a `"models:/"` scheme (e.g., `"models:/iris_model@champion"`).
            dst_name: The name of the registered model to copy the model version to. If a
                registered model with this name does not exist, it will be created.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
            the copied model version.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print(f"Name: {mv.name}")
                print(f"Version: {mv.version}")
                print(f"Source: {mv.source}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

            # Log a model
            with mlflow.start_run() as run:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Create source model version
            client = MlflowClient()
            src_name = "RandomForestRegression-staging"
            client.create_registered_model(src_name)
            src_uri = f"runs:/{run.info.run_id}/sklearn-model"
            mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
            print_model_version_info(mv_src)
            print("--")

            # Copy the source model version into a new registered model
            dst_name = "RandomForestRegression-production"
            src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
            mv_copy = client.copy_model_version(src_model_uri, dst_name)
            print_model_version_info(mv_copy)

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression-staging
            Version: 1
            Source: runs:/53e08bb38f0c487fa36c5872515ed998/sklearn-model
            --
            Name: RandomForestRegression-production
            Version: 1
            Source: models:/RandomForestRegression-staging/1
        """
        if urllib.parse.urlparse(src_model_uri).scheme != "models":
            raise MlflowException(
                f"Unsupported source model URI: '{src_model_uri}'. The `copy_model_version` API "
                "only copies models stored in the 'models:/' scheme."
            )
        client = self._get_registry_client()
        try:
            src_name, src_version = get_model_name_and_version(client, src_model_uri)
            src_mv = client.get_model_version(src_name, src_version)
        except MlflowException as e:
            raise MlflowException(
                f"Failed to fetch model version from source model URI: '{src_model_uri}'. "
                f"Error: {e}"
            ) from e

        if has_prompt_tag(src_mv._tags):
            # Prompt should not be used as a model version
            raise MlflowException(
                f"Model with uri '{src_model_uri}' not found",
                RESOURCE_DOES_NOT_EXIST,
            )

        return client.copy_model_version(src_mv=src_mv, dst_name=dst_name)

    def update_model_version(
        self, name: str, version: str, description: Optional[str] = None
    ) -> ModelVersion:
        """
        Update metadata associated with a model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.
            description: New description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

         .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print(f"Name: {mv.name}")
                print(f"Version: {mv.version}")
                print(f"Description: {mv.description}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)
            # Create a new version of the rfr model under the registered model name
            model_uri = f"runs:/{run.info.run_id}/sklearn-model"
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            print_model_version_info(mv)
            print("--")
            # Update model version's description
            desc = "A new version of the model using ensemble trees"
            mv = client.update_model_version(name, mv.version, desc)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Description: None
            --
            Name: RandomForestRegression
            Version: 1
            Description: A new version of the model using ensemble trees
        """
        if description is None:
            raise MlflowException("Attempting to update model version with no new field values.")

        self._raise_if_prompt(name)
        return self._get_registry_client().update_model_version(
            name=name, version=version, description=description
        )

    @deprecated(since="2.9.0", impact=_STAGES_DEPRECATION_WARNING)
    def transition_model_version_stage(
        self, name: str, version: str, stage: str, archive_existing_versions: bool = False
    ) -> ModelVersion:
        """
        Update model version stage.

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

         .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print(f"Name: {mv.name}")
                print(f"Version: {mv.version}")
                print(f"Description: {mv.description}")
                print(f"Stage: {mv.current_stage}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            desc = "A new version of the model using ensemble trees"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            model_uri = f"runs:/{run.info.run_id}/sklearn-model"
            mv = client.create_model_version(name, model_uri, run.info.run_id, description=desc)
            print_model_version_info(mv)
            print("--")
            # transition model version from None -> staging
            mv = client.transition_model_version_stage(name, mv.version, "staging")
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Description: A new version of the model using ensemble trees
            Stage: None
            --
            Name: RandomForestRegression
            Version: 1
            Description: A new version of the model using ensemble trees
            Stage: Staging
        """
        self._raise_if_prompt(name)
        return self._get_registry_client().transition_model_version_stage(
            name, version, stage, archive_existing_versions
        )

    def delete_model_version(self, name: str, version: str) -> None:
        """
        Delete model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_models_info(mv):
                for m in mv:
                    print(f"name: {m.name}")
                    print(f"latest version: {m.version}")
                    print(f"run_id: {m.run_id}")
                    print(f"current_stage: {m.current_stage}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

            # Create two runs and log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = f"runs:/{run_id}/sklearn-model"
                mv = client.create_model_version(name, model_uri, run_id)
                print(f"model version {mv.version} created")

            print("--")

            # Fetch latest version; this will be version 2
            models = client.get_latest_versions(name, stages=["None"])
            print_models_info(models)
            print("--")

            # Delete the latest model version 2
            print(f"Deleting model version {mv.version}")
            client.delete_model_version(name, mv.version)
            models = client.get_latest_versions(name, stages=["None"])
            print_models_info(models)

        .. code-block:: text
            :caption: Output

            model version 1 created
            model version 2 created
            --
            name: RandomForestRegression
            latest version: 2
            run_id: 9881172ef10f4cb08df3ed452c0c362b
            current_stage: None
            --
            Deleting model version 2
            name: RandomForestRegression
            latest version: 1
            run_id: 9165d4f8aa0a4d069550824bdc55caaf
            current_stage: None
        """
        self._raise_if_prompt(name)
        self._get_registry_client().delete_model_version(name, version)

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """
        Converts the docstring args and returns to google style.

        Args:
            name: Name of the containing registered model.
            version: Version number as an integer of the model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor

            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

            # Create two runs Log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit(X, y)
                signature = infer_signature(X, rfr.predict(X))
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = f"runs:/{run_id}/sklearn-model"
                mv = client.create_model_version(name, model_uri, run_id)
                print(f"model version {mv.version} created")
            print("--")

            # Fetch the last version; this will be version 2
            mv = client.get_model_version(name, mv.version)
            print(f"Name: {mv.name}")
            print(f"Version: {mv.version}")

        .. code-block:: text
            :caption: Output

            model version 1 created
            model version 2 created
            --
            Name: RandomForestRegression
            Version: 2

        """
        mv = self._get_registry_client().get_model_version(name, version)
        if has_prompt_tag(mv._tags):
            raise _model_not_found(name)
        return mv

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """
        Get the download location in Model Registry for this model version.

        Args:
            name: Name of the containing registered model.
            version: Version number as an integer of the model version.

        Returns:
            A single URI location that allows reads for downloading.

        .. code-block:: python

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            model_uri = f"runs:/{run.info.run_id}/sklearn-model"
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            artifact_uri = client.get_model_version_download_uri(name, mv.version)
            print(f"Download URI: {artifact_uri}")

        .. code-block:: text

            Download URI: runs:/027d7bbe81924c5a82b3e4ce979fcab7/sklearn-model
        """
        self._raise_if_prompt(name)
        return self._get_registry_client().get_model_version_download_uri(name, version)

    def search_model_versions(
        self,
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[ModelVersion]:
        """
        Search for model versions in backend that satisfy the filter criteria.

        .. warning:
            The model version search results may not have aliases populated for performance reasons.

        Args:
            filter_string: Filter query string
                (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
                defaults to searching for all model versions. The following identifiers,
                comparators, and logical operators are supported.

                Identifiers
                  - ``name``: model name.
                  - ``source_path``: model version source path.
                  - ``run_id``: The id of the mlflow run that generates the model version.
                  - ``tags.<tag_key>``: model version tag. If ``tag_key`` contains spaces, it must
                    be wrapped with backticks (e.g., ``"tags.`extra key`"``).

                Comparators
                  - ``=``: Equal to.
                  - ``!=``: Not equal to.
                  - ``LIKE``: Case-sensitive pattern match.
                  - ``ILIKE``: Case-insensitive pattern match.
                  - ``IN``: In a value list. Only ``run_id`` identifier supports ``IN`` comparator.

                Logical operators
                  - ``AND``: Combines two sub-queries and returns True if both of them are True.

            max_results: Maximum number of model versions desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_model_versions`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects that satisfy the search expressions. The pagination token for the next
            page can be obtained via the ``token`` attribute of the object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient()

            # Get all versions of the model filtered by name
            model_name = "CordobaWeatherForecastModel"
            filter_string = f"name='{model_name}'"
            results = client.search_model_versions(filter_string)
            print("-" * 80)
            for res in results:
                print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

            # Get the version of the model filtered by run_id
            run_id = "e14afa2f47a040728060c1699968fd43"
            filter_string = f"run_id='{run_id}'"
            results = client.search_model_versions(filter_string)
            print("-" * 80)
            for res in results:
                print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

        .. code-block:: text
            :caption: Output

            ------------------------------------------------------------------------------------
            name=CordobaWeatherForecastModel; run_id=eaef868ee3d14d10b4299c4c81ba8814; version=1
            name=CordobaWeatherForecastModel; run_id=e14afa2f47a040728060c1699968fd43; version=2
            ------------------------------------------------------------------------------------
            name=CordobaWeatherForecastModel; run_id=e14afa2f47a040728060c1699968fd43; version=2
        """
        return self._get_registry_client().search_model_versions(
            filter_string, max_results, order_by, page_token
        )

    @deprecated(since="2.9.0", impact=_STAGES_DEPRECATION_WARNING)
    def get_model_version_stages(self, name: str, version: str) -> list[str]:
        """
        This is a docstring. Here is info.

        Returns:
            A list of valid stages.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # fetch valid stages
            model_uri = f"runs:/{run.info.run_id}/models/sklearn-model"
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            stages = client.get_model_version_stages(name, mv.version)
            print(f"Model list of valid stages: {stages}")

        .. code-block:: text
            :caption: Output

            Model list of valid stages: ['None', 'Staging', 'Production', 'Archived']

        """
        return ALL_STAGES

    def set_model_version_tag(
        self,
        name: str,
        version: Optional[str] = None,
        key: Optional[str] = None,
        value: Any = None,
        stage: Optional[str] = None,
    ) -> None:
        """Set a tag for the model version.
        When stage is set, tag will be set for latest model version of the stage.
        Setting both version and stage parameter will result in error.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key to log. key is required.
            value: Tag value to log. value is required.
            stage: Registered model stage.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print(f"Name: {mv.name}")
                print(f"Version: {mv.version}")
                print(f"Tags: {mv.tags}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # and set a tag
            model_uri = f"runs:/{run.info.run_id}/sklearn-model"
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            print_model_version_info(mv)
            print("--")

            # Tag using model version
            client.set_model_version_tag(name, mv.version, "t", "1")

            # Tag using model stage
            client.set_model_version_tag(name, key="t1", value="1", stage=mv.current_stage)
            mv = client.get_model_version(name, mv.version)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Tags: {}
            --
            Name: RandomForestRegression
            Version: 1
            Tags: {'t': '1', 't1': '1'}
        """
        _validate_model_version_or_stage_exists(version, stage)
        self._raise_if_prompt(name)
        if stage:
            warnings.warn(
                "The `stage` parameter of the `set_model_version_tag` API is deprecated. "
                + _STAGES_DEPRECATION_WARNING,
                category=FutureWarning,
                stacklevel=2,
            )
            latest_versions = self.get_latest_versions(name, stages=[stage])
            if not latest_versions:
                raise MlflowException(f"Could not find any model version for {stage} stage")
            version = latest_versions[0].version

        self._get_registry_client().set_model_version_tag(name, version, key, value)

    def delete_model_version_tag(
        self,
        name: str,
        version: Optional[str] = None,
        key: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Delete a tag associated with the model version.

        When stage is set, tag will be deleted for latest model version of the stage.
        Setting both version and stage parameter will result in error.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key. key is required.
            stage: Registered model stage.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print(f"Name: {mv.name}")
                print(f"Version: {mv.version}")
                print(f"Tags: {mv.tags}")


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)
            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # and delete a tag
            model_uri = f"runs:/{run.info.run_id}/sklearn-model"
            tags = {"t": "1", "t1": "2"}
            mv = client.create_model_version(name, model_uri, run.info.run_id, tags=tags)
            print_model_version_info(mv)
            print("--")
            # using version to delete tag
            client.delete_model_version_tag(name, mv.version, "t")

            # using stage to delete tag
            client.delete_model_version_tag(name, key="t1", stage=mv.current_stage)
            mv = client.get_model_version(name, mv.version)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Tags: {'t': '1', 't1': '2'}
            --
            Name: RandomForestRegression
            Version: 1
            Tags: {}
        """
        _validate_model_version_or_stage_exists(version, stage)
        self._raise_if_prompt(name)
        if stage:
            warnings.warn(
                "The `stage` parameter of the `delete_model_version_tag` API is deprecated. "
                + _STAGES_DEPRECATION_WARNING,
                category=FutureWarning,
                stacklevel=2,
            )
            latest_versions = self.get_latest_versions(name, stages=[stage])
            if not latest_versions:
                raise MlflowException(f"Could not find any model version for {stage} stage")
            version = latest_versions[0].version
        self._get_registry_client().delete_model_version_tag(name, version, key)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias. Note that aliases of the format ``v<number>``, such as
                ``v9`` and ``v42``, are reserved and cannot be set.
            version: Registered model version number.

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_info(rm):
                print("--Model--")
                print("name: {}".format(rm.name))
                print("aliases: {}".format(rm.aliases))


            def print_model_version_info(mv):
                print("--Model Version--")
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Aliases: {}".format(mv.aliases))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)
            model = client.get_registered_model(name)
            print_model_info(model)

            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            print_model_version_info(mv)

            # Set registered model alias
            client.set_registered_model_alias(name, "test-alias", mv.version)
            print()
            print_model_info(model)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            --Model--
            name: RandomForestRegression
            aliases: {}

            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: []

            --Model--
            name: RandomForestRegression
            aliases: {"test-alias": "1"}

            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: ["test-alias"]
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        _validate_model_version(version)
        self._raise_if_prompt(name)
        self._get_registry_client().set_registered_model_alias(name, alias, version)

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        """Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_info(rm):
                print("--Model--")
                print("name: {}".format(rm.name))
                print("aliases: {}".format(rm.aliases))


            def print_model_version_info(mv):
                print("--Model Version--")
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Aliases: {}".format(mv.aliases))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)
            model = client.get_registered_model(name)
            print_model_info(model)

            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            print_model_version_info(mv)

            # Set registered model alias
            client.set_registered_model_alias(name, "test-alias", mv.version)
            print()
            print_model_info(model)
            print_model_version_info(mv)

            # Delete registered model alias
            client.delete_registered_model_alias(name, "test-alias")
            print()
            print_model_info(model)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            --Model--
            name: RandomForestRegression
            aliases: {}
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: []

            --Model--
            name: RandomForestRegression
            aliases: {"test-alias": "1"}
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: ["test-alias"]

            --Model--
            name: RandomForestRegression
            aliases: {}
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: []
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        self._raise_if_prompt(name)
        self._get_registry_client().delete_registered_model_alias(name, alias)

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        """Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

         .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
            from mlflow.models import infer_signature
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor


            def print_model_info(rm):
                print("--Model--")
                print("name: {}".format(rm.name))
                print("aliases: {}".format(rm.aliases))


            def print_model_version_info(mv):
                print("--Model Version--")
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Aliases: {}".format(mv.aliases))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))
            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)
            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)
            model = client.get_registered_model(name)
            print_model_info(model)
            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            print_model_version_info(mv)
            # Set registered model alias
            client.set_registered_model_alias(name, "test-alias", mv.version)
            print()
            print_model_info(model)
            print_model_version_info(mv)
            # Get model version by alias
            alias_mv = client.get_model_version_by_alias(name, "test-alias")
            print()
            print_model_version_info(alias_mv)

        .. code-block:: text
            :caption: Output

            --Model--
            name: RandomForestRegression
            aliases: {}
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: []
            --Model--
            name: RandomForestRegression
            aliases: {"test-alias": "1"}
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: ["test-alias"]
            --Model Version--
            Name: RandomForestRegression
            Version: 1
            Aliases: ["test-alias"]
        """
        _validate_model_name(name)
        mv = self._get_registry_client().get_model_version_by_alias(name, alias)

        if has_prompt_tag(mv._tags):
            raise _model_not_found(name)

        return mv

    def _raise_if_prompt(self, name: str) -> None:
        """
        Validate if the given name is registered as a Prompt rather than a Registered Model.
        """
        rm = self.get_registered_model(name)
        if has_prompt_tag(rm._tags):
            raise _model_not_found(name)

    @experimental
    def create_logged_model(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        source_run_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        params: Optional[dict[str, str]] = None,
        model_type: Optional[str] = None,
    ) -> LoggedModel:
        """
        Create a new logged model.

        Args:
            experiment_id: ID of the experiment to which the model belongs.
            name: Name of the model. If not specified, a random name will be generated.
            source_run_id: ID of the run that produced the model.
            tags: Tags to set on the model.
            params: Parameters to set on the model.
            model_type: The type of the model. This is a user-defined string that can be used to
                        search and compare related models. For example, setting
                        ``model_type="agent"`` enables you to easily search for this model and
                        compare it to other models of type ``"agent"`` in the future.

        Returns:
            The created model.
        """
        return self._tracking_client.create_logged_model(
            experiment_id, name, source_run_id, tags, params, model_type
        )

    @experimental
    def log_model_params(self, model_id: str, params: dict[str, str]) -> None:
        """
        Log parameters for a logged model.

        Args:
            model_id: ID of the model to log parameters for.
            params: Dictionary of parameters to log.

        Returns:
            None
        """
        _validate_model_id_specified(model_id)
        try:
            self._tracking_client.get_logged_model(model_id)
        except Exception as e:
            extra_error_reason = (
                "the logged model may exist in a different workspace, "
                if is_in_databricks_runtime()
                else ""
            )
            raise MlflowException.invalid_parameter_value(
                f"Failed to get logged model with ID {model_id}. "
                f"You may not have access to the logged model, {extra_error_reason}or the "
                "logged model may have been deleted. If you are attempting to log parameters "
                "to a Registered Model Version, this is not supported."
            ) from e
        return self._tracking_client.log_model_params(model_id, params)

    @experimental
    def finalize_logged_model(
        self, model_id: str, status: Union[Literal["READY", "FAILED"], LoggedModelStatus]
    ) -> LoggedModel:
        """
        Finalize a model by updating its status.

        Args:
            model_id: ID of the model to finalize.
            status: Final status to set on the model.

        Returns:
            The updated model.
        """
        _validate_model_id_specified(model_id)
        return self._tracking_client.finalize_logged_model(
            model_id, LoggedModelStatus(status) if isinstance(status, str) else status
        )

    @experimental
    def get_logged_model(self, model_id: str) -> LoggedModel:
        """
        Fetch the logged model with the specified ID.

        Args:
            model_id: ID of the model to fetch.

        Returns:
            The fetched model.
        """
        _validate_model_id_specified(model_id)
        return self._tracking_client.get_logged_model(model_id)

    @experimental
    def delete_logged_model(self, model_id: str) -> None:
        """
        Delete the logged model with the specified ID.

        Args:
            model_id: ID of the model to delete.
        """
        _validate_model_id_specified(model_id)
        return self._tracking_client.delete_logged_model(model_id)

    @experimental
    def set_logged_model_tags(self, model_id: str, tags: dict[str, Any]) -> None:
        """
        Set tags on the specified logged model.

        Args:
            model_id: ID of the model.
            tags: Tags to set on the model.

        Returns:
            None
        """
        _validate_model_id_specified(model_id)
        try:
            self._tracking_client.get_logged_model(model_id)
        except Exception as e:
            extra_error_reason = (
                "the logged model may exist in a different workspace, "
                if is_in_databricks_runtime()
                else ""
            )
            raise MlflowException.invalid_parameter_value(
                f"Failed to get logged model with ID {model_id}. "
                f"You may not have access to the logged model, {extra_error_reason}or the "
                "logged model may have been deleted. If you are attempting to "
                "set tags on a Registered Model Version, use `set_model_version_tag` instead."
            ) from e
        self._tracking_client.set_logged_model_tags(model_id, tags)

    @experimental
    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """
        Delete a tag from the specified logged model.

        Args:
            model_id: ID of the model.
            key: Tag key to delete.

        """
        _validate_model_id_specified(model_id)
        return self._tracking_client.delete_logged_model_tag(model_id, key)

    def log_model_artifact(self, model_id: str, local_path: str) -> None:
        """
        Upload an artifact to the specified logged model.

        Args:
            model_id: ID of the model.
            local_path: Local path to the artifact to upload.

        Returns:
            None
        """
        return self._tracking_client.log_model_artifact(model_id, local_path)

    def log_model_artifacts(self, model_id: str, local_dir: str) -> None:
        """
        Upload a set of artifacts to the specified logged model.

        Args:
            model_id: ID of the model.
            local_dir: Local directory containing the artifacts to upload.

        Returns:
            None
        """
        return self._tracking_client.log_model_artifacts(model_id, local_dir)

    @experimental
    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        datasets: Optional[list[dict[str, Any]]] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[dict[str, Any]]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[LoggedModel]:
        """
        Search for logged models that match the specified search criteria.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A SQL-like filter string to parse. The filter string syntax supports:

                - Entity specification:
                    - attributes: `attribute_name` (default if no prefix is specified)
                    - metrics: `metrics.metric_name`
                    - parameters: `params.param_name`
                    - tags: `tags.tag_name`
                - Comparison operators:
                    - For numeric entities (metrics and numeric attributes): <, <=, >, >=, =, !=
                    - For string entities (params, tags, string attributes): =, !=, IN, NOT IN
                - Multiple conditions can be joined with 'AND'
                - String values must be enclosed in single quotes

                Example filter strings:
                    - `creation_time > 100`
                    - `metrics.rmse > 0.5 AND params.model_type = 'rf'`
                    - `tags.release IN ('v1.0', 'v1.1')`
                    - `params.optimizer != 'adam' AND metrics.accuracy >= 0.9`
            datasets: List of dictionaries to specify datasets on which to apply metrics filters
                For example, a filter string with `metrics.accuracy > 0.9` and dataset with name
                "test_dataset" means we will return all logged models with accuracy > 0.9 on the
                test_dataset. Metric values from ANY dataset matching the criteria are considered.
                If no datasets are specified, then metrics across all datasets are considered in
                the filter. The following fields are supported:

                dataset_name (str):
                    Required. Name of the dataset.
                dataset_digest (str):
                    Optional. Digest of the dataset.
            max_results: Maximum number of logged models desired.
            order_by: List of dictionaries to specify the ordering of the search results.
                The following fields are supported:

                field_name (str):
                    Required. Name of the field to order by, e.g. "metrics.accuracy".
                ascending (bool):
                    Optional. Whether the order is ascending or not.
                dataset_name (str):
                    Optional. If ``field_name`` refers to a metric, this field
                    specifies the name of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name will be considered for ordering.
                    This field may only be set if ``field_name`` refers to a metric.
                dataset_digest (str):
                    Optional. If ``field_name`` refers to a metric, this field
                    specifies the digest of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name and digest will be considered for
                    ordering. This field may only be set if ``dataset_name`` is also set.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`LoggedModel <mlflow.entities.LoggedModel>` objects.
        """
        return self._tracking_client.search_logged_models(
            experiment_ids, filter_string, datasets, max_results, order_by, page_token
        )

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def create_prompt(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
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
            A Prompt object.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            prompt_info = client.create_prompt(
                name="my_prompt",
                description="A helpful prompt",
                tags={"team": "data-science"},
            )
        """
        registry_client = self._get_registry_client()
        return registry_client.create_prompt(name, description, tags)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def create_prompt_version(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> PromptVersion:
        """
        Create a new version of an existing prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            template: Template text of the prompt version.
            description: Optional description of the prompt version.
            tags: Optional dictionary of prompt version tags.

        Returns:
            A PromptVersion object.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            prompt_version = client.create_prompt_version(
                name="my_prompt",
                template="Respond as a {{style}} assistant: {{query}}",
                description="Added style parameter",
                tags={"author": "alice"},
            )
        """
        registry_client = self._get_registry_client()
        return registry_client.create_prompt_version(name, template, description, tags)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def get_prompt(self, name: str) -> Optional[Prompt]:
        """
        Get prompt metadata by name.

        Args:
            name: Name of the prompt.

        Returns:
            A Prompt object containing prompt metadata, or None if not found.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            prompt = client.get_prompt("my_prompt")
            if prompt:
                print(f"Prompt: {prompt.name}")
                print(f"Description: {prompt.description}")
        """
        registry_client = self._get_registry_client()
        return registry_client.get_prompt(name)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def get_prompt_version(self, name: str, version: Union[str, int]) -> Optional[PromptVersion]:
        """
        Get a specific prompt version.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version of the prompt (number or alias).

        Returns:
            A PromptVersion object with the specific version content, or None if not found.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            prompt_version = client.get_prompt_version("my_prompt", "1")
            prompt_alias = client.get_prompt_version("my_prompt", "production")
        """
        registry_client = self._get_registry_client()
        return registry_client.get_prompt_version(name, version)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def delete_prompt_version(self, name: str, version: str) -> None:
        """
        Delete a specific prompt version.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to delete.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            client.delete_prompt_version("my_prompt", "1")
        """
        registry_client = self._get_registry_client()
        return registry_client.delete_prompt_version(name, version)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            key: Tag key.
            value: Tag value.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            client.set_prompt_tag("my_prompt", "environment", "production")
        """
        registry_client = self._get_registry_client()
        return registry_client.set_prompt_tag(name, key, value)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            key: Tag key to delete.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            client.delete_prompt_tag("my_prompt", "environment")
        """
        registry_client = self._get_registry_client()
        return registry_client.delete_prompt_tag(name, key)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def get_prompt_version_by_alias(self, name: str, alias: str) -> PromptVersion:
        """
        Get a prompt version by alias.

        This method delegates directly to the store, providing full Unity Catalog support
        when used with Unity Catalog registries.

        Args:
            name: Name of the prompt.
            alias: Alias of the prompt version.

        Returns:
            A PromptVersion object.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            prompt_version = client.get_prompt_version_by_alias("my_prompt", "production")
        """
        registry_client = self._get_registry_client()
        return registry_client.get_prompt_version_by_alias(name, alias)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def search_prompt_versions(
        self, name: str, max_results: Optional[int] = None, page_token: Optional[str] = None
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

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()
            response = client.search_prompt_versions("my_prompt", max_results=10)
            for version in response.prompt_versions:
                print(f"Version {version.version}: {version.description}")
        """
        registry_client = self._get_registry_client()
        return registry_client.search_prompt_versions(name, max_results, page_token)

    @experimental
    @require_prompt_registry
    @translate_prompt_exception
    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from the registry.

        For Unity Catalog registries, this method first checks if any versions exist for the prompt
        and throws an error if undeleted versions are found. All versions must be explicitly
        deleted first before the prompt itself can be deleted.

        For other registries, the prompt is deleted normally without version checking.

        Args:
            name: Name of the prompt to delete.

        Example:

        .. code-block:: python

            from mlflow import MlflowClient

            client = MlflowClient()

            # For Unity Catalog, delete all versions first
            if client.get_registry_uri().startswith("databricks-uc"):
                versions = client.search_prompt_versions("my_prompt")
                for version in versions.prompt_versions:
                    client.delete_prompt_version("my_prompt", version.version)

            # Then delete the prompt
            client.delete_prompt("my_prompt")
        """
        registry_client = self._get_registry_client()

        # Only check for existing versions in Unity Catalog registries
        registry_uri = self._registry_uri

        if is_databricks_unity_catalog_uri(registry_uri):
            search_response = self.search_prompt_versions(name, max_results=1)

            # Check if any versions exist
            has_versions = (
                hasattr(search_response, "prompt_versions")
                and search_response.prompt_versions
                and len(search_response.prompt_versions) > 0
            )

            if has_versions:
                raise MlflowException(
                    f"Cannot delete prompt '{name}' because it still has undeleted versions. "
                    f"Please delete all versions first using delete_prompt_version(), "
                    f"then delete the prompt.",
                    INVALID_PARAMETER_VALUE,
                )

        # For non-Unity Catalog registries, or if version check passes, delete the prompt
        return registry_client.delete_prompt(name)
