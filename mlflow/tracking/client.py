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
from mlflow.entities.model_registry import ModelVersion, Prompt, RegisteredModel
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID, NoOpSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import (
    IS_PROMPT_TAG_KEY,
    PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
    PROMPT_NAME_RULE,
)
from mlflow.prompt.registry_utils import (
    has_prompt_tag,
    translate_prompt_exception,
    validate_prompt_name,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    FEATURE_DISABLED,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
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


def _parse_prompt_uri(uri: str) -> tuple[str, Optional[str]]:
    """
    Parse prompt URI into prompt name and prompt version.
    - 'prompts:/<name>/<version>' -> ('<name>', '<version>')
    - 'prompts:/<name>@<alias>' -> ('<name>', '<alias>')
    - 'prompts:/<name>' -> ('<name>', None)
    """
    parsed = urllib.parse.urlparse(uri)
    
    if parsed.scheme != "prompts":
        raise MlflowException.invalid_parameter_value(
            f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
        )
    
    path = parsed.path.lstrip("/")
    
    # Check for alias format (name@alias)
    if "@" in path:
        name, alias = path.split("@", 1)
        return name, alias
    
    # Check for version format (name/version)
    parts = path.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    # Just name
    return parts[0], None


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

    def set_terminated(self, run_id: str, status: Optional[str] = None, end_time: Optional[int] = None) -> None:
        """
        Set a run's status to terminated.

        Args:
            run_id: Unique identifier for the run.
            status: Updated status of the run. Defaults to "FINISHED".
            end_time: Unix timestamp of when the run ended, in milliseconds.

        Returns:
            None
        """
        return self._tracking_client.set_terminated(run_id, status, end_time)

    ##### Prompt Registry #####

    @experimental
    @translate_prompt_exception
    def register_prompt(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Prompt:
        """
        Register a :py:class:`Prompt <mlflow.entities.Prompt>` with the MLflow Prompt Registry.

        Args:
            name: The name of the prompt.
            template: The template text of the prompt. It can contain variables enclosed in
                double curly braces, e.g. {{variable}}, which will be replaced with actual values
                by the `format` method. MLflow use the same variable naming rules same as Jinja2
                https://jinja.palletsprojects.com/en/stable/api/#notes-on-identifiers
            description: A description of the prompt. Optional.
            tags: A dictionary of tags to associate with the prompt. Optional.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            prompt = client.register_prompt(
                name="my_prompt",
                template="Hello, {{name}}!",
                description="A simple greeting prompt",
                tags={"category": "greeting"},
            )

        """
        registry_client = self._get_registry_client()
        return registry_client.create_prompt(name, template, description, tags)

    @experimental
    @translate_prompt_exception
    def load_prompt(
        self, name_or_uri: str, version: Optional[str] = None, allow_missing: bool = False
    ) -> Prompt:
        """
        Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

        The prompt can be specified by name and version, or by URI.

        Args:
            name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
            version: The version of the prompt. If not specified, the latest version will be loaded.
            allow_missing: If True, return None instead of raising Exception if the specified prompt
                is not found.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()

            # Load the latest version of the prompt by name
            prompt = client.load_prompt("my_prompt")

            # Load a specific version of the prompt by name and version
            prompt = client.load_prompt("my_prompt", version="1")

            # Load a specific version of the prompt by URI
            prompt = client.load_prompt("prompts:/my_prompt/1")

            # Load a prompt version with an alias "production"
            prompt = client.load_prompt("prompts:/my_prompt@production")

        """
        # Parse URI if provided
        if name_or_uri.startswith("prompts:/"):
            name, version = _parse_prompt_uri(name_or_uri)
        else:
            name = name_or_uri

        registry_client = self._get_registry_client()
        prompt = registry_client.get_prompt(name, version)

        if prompt is None:
            if allow_missing:
                return None
            version_str = version if version is not None else "latest"
            raise MlflowException(
                f"Prompt with name={name}, version={version_str} not found",
                RESOURCE_DOES_NOT_EXIST,
            )

        return prompt

    @experimental
    @translate_prompt_exception
    def search_prompts(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in the MLflow Prompt Registry.

        Args:
            filter_string: Filter query string, defaults to searching all prompts.
                Currently supports a single filter condition like "name = 'my_prompt'".
            max_results: Maximum number of prompts desired. Default is 100.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_prompts`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Prompt <mlflow.entities.Prompt>` objects that satisfy the search
            expressions. The pagination token for the next page can be obtained via the
            ``token`` attribute of the object.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()

            # Search for all prompts
            prompts = client.search_prompts()

            # Search for prompts with a specific name
            prompts = client.search_prompts(filter_string="name = 'my_prompt'")

        """
        registry_client = self._get_registry_client()
        return registry_client.search_prompts(filter_string, max_results, order_by, page_token)

    @experimental
    @translate_prompt_exception
    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from the MLflow Prompt Registry.

        Args:
            name: The name of the prompt to delete.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.delete_prompt("my_prompt")

        """
        registry_client = self._get_registry_client()
        registry_client.delete_prompt(name)

    @experimental
    @translate_prompt_exception
    def create_prompt_version(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Prompt:
        """
        Create a new version of an existing prompt.

        Args:
            name: The name of the prompt.
            template: The template text for this version.
            description: A description of this version. Optional.
            tags: A dictionary of tags to associate with this version. Optional.

        Returns:
            A :py:class:`Prompt <mlflow.entities.Prompt>` object representing the new version.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            prompt = client.create_prompt_version(
                name="my_prompt",
                template="Hello, {{title}} {{name}}!",
                description="Added title field",
            )

        """
        registry_client = self._get_registry_client()
        return registry_client.create_prompt_version(name, template, description, tags)

    @experimental
    @translate_prompt_exception
    def get_prompt_version(self, name: str, version: str) -> Prompt:
        """
        Get a specific version of a prompt.

        Args:
            name: The name of the prompt.
            version: The version number of the prompt.

        Returns:
            A :py:class:`Prompt <mlflow.entities.Prompt>` object.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            prompt = client.get_prompt_version("my_prompt", "2")

        """
        registry_client = self._get_registry_client()
        return registry_client.get_prompt_version(name, version)

    @experimental
    @translate_prompt_exception
    def delete_prompt_version(self, name: str, version: str) -> None:
        """
        Delete a specific version of a prompt.

        Args:
            name: The name of the prompt.
            version: The version number to delete.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.delete_prompt_version("my_prompt", "2")

        """
        registry_client = self._get_registry_client()
        registry_client.delete_prompt_version(name, version)

    @experimental
    @translate_prompt_exception
    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt.

        Args:
            name: The name of the prompt.
            key: The tag key.
            value: The tag value.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.set_prompt_tag("my_prompt", "category", "greeting")

        """
        registry_client = self._get_registry_client()
        registry_client.set_prompt_tag(name, key, value)

    @experimental
    @translate_prompt_exception
    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt.

        Args:
            name: The name of the prompt.
            key: The tag key to delete.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.delete_prompt_tag("my_prompt", "category")

        """
        registry_client = self._get_registry_client()
        registry_client.delete_prompt_tag(name, key)

    @experimental
    @translate_prompt_exception
    def set_prompt_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt version.

        Args:
            name: The name of the prompt.
            version: The version number.
            key: The tag key.
            value: The tag value.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.set_prompt_version_tag("my_prompt", "1", "status", "tested")

        """
        registry_client = self._get_registry_client()
        registry_client.set_prompt_version_tag(name, version, key, value)

    @experimental
    @translate_prompt_exception
    def delete_prompt_version_tag(self, name: str, version: str, key: str) -> None:
        """
        Delete a tag from a prompt version.

        Args:
            name: The name of the prompt.
            version: The version number.
            key: The tag key to delete.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.delete_prompt_version_tag("my_prompt", "1", "status")

        """
        registry_client = self._get_registry_client()
        registry_client.delete_prompt_version_tag(name, version, key)

    @experimental
    @translate_prompt_exception
    def set_prompt_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set an alias for a prompt version.

        Args:
            name: The name of the prompt.
            alias: The alias to set.
            version: The version to alias.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.set_prompt_alias("my_prompt", "production", "3")

        """
        registry_client = self._get_registry_client()
        registry_client.set_prompt_alias(name, alias, version)

    @experimental
    @translate_prompt_exception
    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete a prompt alias.

        Args:
            name: The name of the prompt.
            alias: The alias to delete.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            client.delete_prompt_alias("my_prompt", "production")

        """
        registry_client = self._get_registry_client()
        registry_client.delete_prompt_alias(name, alias)

    @experimental
    @translate_prompt_exception
    def get_prompt_version_by_alias(self, name: str, alias: str) -> Prompt:
        """
        Get a prompt version by alias.

        Args:
            name: The name of the prompt.
            alias: The alias to look up.

        Returns:
            A :py:class:`Prompt <mlflow.entities.Prompt>` object.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()
            prompt = client.get_prompt_version_by_alias("my_prompt", "production")

        """
        registry_client = self._get_registry_client()
        return registry_client.get_prompt_version_by_alias(name, alias)

    @experimental
    @translate_prompt_exception
    def log_prompt(self, run_id: str, prompt: Union[str, Prompt]) -> None:
        """
        Log a prompt to an MLflow run.

        Args:
            run_id: The ID of the run to log the prompt to.
            prompt: The prompt to log. Can be either a prompt URI (e.g. "prompts:/my_prompt/1")
                or a Prompt object.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()

            # Log a prompt by URI
            with mlflow.start_run() as run:
                client.log_prompt(run.info.run_id, "prompts:/my_prompt/1")

            # Log a prompt object
            prompt = client.load_prompt("my_prompt")
            with mlflow.start_run() as run:
                client.log_prompt(run.info.run_id, prompt)

        """
        if isinstance(prompt, str):
            # Parse prompt URI
            name, version = _parse_prompt_uri(prompt)
            prompt_obj = self.load_prompt(name, version)
            if prompt_obj is None:
                raise MlflowException(
                    f"Prompt '{prompt}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
        else:
            prompt_obj = prompt

        # Get current tags
        run = self.get_run(run_id)
        current_prompts = run.data.tags.get(PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, "")
        
        # Add prompt URI to the list
        prompt_uri = f"prompts:/{prompt_obj.name}/{prompt_obj.version}"
        if current_prompts:
            prompt_list = current_prompts.split(",")
            if prompt_uri not in prompt_list:
                prompt_list.append(prompt_uri)
                new_prompts = ",".join(prompt_list)
            else:
                return  # Already logged
        else:
            new_prompts = prompt_uri

        # Update the tag
        self.set_tag(run_id, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, new_prompts)

    @experimental
    @translate_prompt_exception
    def detach_prompt_from_run(self, run_id: str, prompt_uri: str) -> None:
        """
        Detach a prompt from an MLflow run.

        Args:
            run_id: The ID of the run to detach the prompt from.
            prompt_uri: The URI of the prompt to detach (e.g. "prompts:/my_prompt/1").

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()

            # Detach a prompt from a run
            with mlflow.start_run() as run:
                client.log_prompt(run.info.run_id, "prompts:/my_prompt/1")
                client.detach_prompt_from_run(run.info.run_id, "prompts:/my_prompt/1")

        """
        # Get current tags
        run = self.get_run(run_id)
        current_prompts = run.data.tags.get(PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, "")
        
        if not current_prompts:
            return  # No prompts to detach
        
        # Remove prompt URI from the list
        prompt_list = current_prompts.split(",")
        if prompt_uri in prompt_list:
            prompt_list.remove(prompt_uri)
            if prompt_list:
                new_prompts = ",".join(prompt_list)
                self.set_tag(run_id, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, new_prompts)
            else:
                self.delete_tag(run_id, PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY)

    @experimental
    @translate_prompt_exception
    def list_logged_prompts(self, run_id: str) -> list[Prompt]:
        """
        List all prompts logged to an MLflow run.

        Args:
            run_id: The ID of the run to list prompts for.

        Returns:
            A list of Prompt objects.

        Example:

        .. code-block:: python

            import mlflow

            client = mlflow.MlflowClient()

            with mlflow.start_run() as run:
                client.log_prompt(run.info.run_id, "prompts:/my_prompt/1")
                client.log_prompt(run.info.run_id, "prompts:/my_prompt/2")
                
                prompts = client.list_logged_prompts(run.info.run_id)
                for prompt in prompts:
                    print(f"{prompt.name} v{prompt.version}")

        """
        # Get current tags
        run = self.get_run(run_id)
        current_prompts = run.data.tags.get(PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY, "")
        
        if not current_prompts:
            return []
        
        # Load each prompt
        prompts = []
        for prompt_uri in current_prompts.split(","):
            try:
                name, version = _parse_prompt_uri(prompt_uri)
                prompt = self.load_prompt(name, version, allow_missing=True)
                if prompt:
                    prompts.append(prompt)
            except Exception:
                # Skip invalid URIs
                pass
        
        return prompts

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
            root_span = client.start_trace("my_trace", tags={"key": "value"})
            client.set_trace_tag(root_span.trace_id, "key", "value")
            client.end_trace(root_span.trace_id)

        Args:
            trace_id: The ID of the trace to set the tag on.
            key: The tag key.
            value: The tag value.
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
            key: The tag key to delete.
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
            print(f"Tags: {experiment.tags}")
            print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

        .. code-block:: text
            :caption: Output

            Name: Social NLP Experiments
            Tags: {'version': 'v1', 'priority': 'P1', 'nlp.framework': 'Spark NLP'}
            Lifecycle_stage: active

        """
        return self._tracking_client.create_experiment(name, artifact_location, tags)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the backend store.

        This deletion is a soft-delete, not a permanent deletion. Experiment names can not
        be reused, unless the deleted experiment is permanently deleted by a database admin.

        Args:
            experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_experiment_info(experiment):
                print(f"Name: {experiment.name}")
                print(f"Artifact Location: {experiment.artifact_location}")
                print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


            # Create an experiment with a name that is unique and case sensitive
            client = MlflowClient()
            experiment_id = client.create_experiment("New Experiment")
            client.delete_experiment(experiment_id)

            # Examine the deleted experiment details.
            experiment = client.get_experiment(experiment_id)
            print_experiment_info(experiment)

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
                All backend stores will support keys up to length 250, but some
                may support larger keys.
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
            # To end the run, you'll have to explicitly terminate it.
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
                All backend stores support keys up to length 250, but some
                may support larger keys.
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

            run_id: 4f226eb5758145e9b28f78514b59a03b
            params: {}
            status: RUNNING
            --
            run_id: 4f226eb5758145e9b28f78514b59a03b
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
            Tags: {'t2': '2'}
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

            from mlflow import MlflowClient


            def print_run_info(r):
                print(f"run_id: {r.info.run_id}")
                print(f"run_name: {r.info.run_name}")
                print(f"status: {r.info.status}")


            # Create a run under the default experiment (whose id is '0').
            # Since these are low-level CRUD operations, this method will create a run.
            # To end the run, you'll have to explicitly terminate it.
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
            status: FINISHED
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

        .. code-block:: text
            :caption: Output

            run_id: ef0247fa32054ee7860a964fad31cb3f
            metrics: {'m': 1.5}
            status: FINISHED
            --
            run_id: ef0247fa32054ee7860a964fad31cb3f
            metrics: {'m': 1.5}
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
                print_artifact_info(artifact)
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: features.txt
            is_dir: False
            size: 53

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
                client.log_artifact(run_id, tmp_dir, "data.json")

            # Log artifacts
            client.log_artifacts(run_id, tmp_dir, "data")

        .. code-block:: text
            :caption: Output

            artifact: data.json
            is_dir: False
            size: 53

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
            run_id: The unique identifier for the run.
            artifact_file: The file path of the artifact relative to the run's directory.
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

            # Log text to a file under the run's root directory
            client.log_text(run.info.run_id, "text1", "file1.txt")

        .. code-block:: text
            :caption: Output

            artifact: text1
            is_dir: False
            size: 5

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

            # Log a dictionary as a JSON file under the run's root directory
            client.log_dict(run_id, dictionary, "data.json")

        .. code-block:: text
            :caption: Output

            artifact: data.json
            is_dir: False
            size: 5

        """
        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            with open(tmp_path, "w") as f:
                # Specify `indent` to prettify the output
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
        """
        List all artifacts for a given run.

        Args:
            run_id: String ID of the run.
            path: The path to list artifacts for. If None, lists all artifacts.

        Returns:
            A list of :py:class:`mlflow.entities.FileInfo` objects.
        """
        return self._tracking_client.list_artifacts(run_id, path)