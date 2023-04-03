"""
Internal package providing a Python CRUD interface to MLflow experiments, runs, registered models,
and model versions. This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module,
and is exposed in the :py:mod:`mlflow.tracking` module.
"""
import contextlib
import logging
import json
import os
import posixpath
import sys
import tempfile
import yaml
from typing import Any, Dict, Sequence, List, Optional, Union, TYPE_CHECKING

from mlflow.entities import Experiment, Run, Param, Metric, RunTag, FileInfo, ViewType
from mlflow.store.entities.paged_list import PagedList
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._model_registry.client import ModelRegistryClient
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.tracking.artifact_utils import _upload_artifacts_to_databricks
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.annotations import deprecated
from mlflow.utils.databricks_utils import get_databricks_run_url
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_databricks_uri, is_databricks_unity_catalog_uri
from mlflow.utils.validation import (
    _validate_model_version_or_stage_exists,
    _validate_model_name,
    _validate_model_alias_name,
    _validate_model_version,
)

if TYPE_CHECKING:
    import matplotlib  # pylint: disable=unused-import
    import plotly  # pylint: disable=unused-import
    import numpy  # pylint: disable=unused-import
    import PIL  # pylint: disable=unused-import

_logger = logging.getLogger(__name__)


class MlflowClient:
    """
    Client of an MLflow Tracking Server that creates and manages experiments and runs, and of an
    MLflow Registry Server that creates and manages registered models and model versions. It's a
    thin wrapper around TrackingServiceClient and RegistryClient so there is a unified API but we
    can keep the implementation of the tracking and registry clients independent from each other.
    """

    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """
        :param tracking_uri: Address of local or remote tracking server. If not provided, defaults
                             to the service set by ``mlflow.tracking.set_tracking_uri``. See
                             `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                             for more info.
        :param registry_uri: Address of local or remote model registry server. If not provided,
                             defaults to the service set by ``mlflow.tracking.set_registry_uri``. If
                             no such service was set, defaults to the tracking uri of the client.
        """
        final_tracking_uri = utils._resolve_tracking_uri(tracking_uri)
        self._registry_uri = registry_utils._resolve_registry_uri(registry_uri, tracking_uri)
        self._tracking_client = TrackingServiceClient(final_tracking_uri)
        # `MlflowClient` also references a `ModelRegistryClient` instance that is provided by the
        # `MlflowClient._get_registry_client()` method. This `ModelRegistryClient` is not explicitly
        # defined as an instance variable in the `MlflowClient` constructor; an instance variable
        # is assigned lazily by `MlflowClient._get_registry_client()` and should not be referenced
        # outside of the `MlflowClient._get_registry_client()` method

    @property
    def tracking_uri(self):
        return self._tracking_client.tracking_uri

    def _get_registry_client(self):
        """
        Attempts to create a py:class:`ModelRegistryClient` if one does not already exist.

        :raises: py:class:`mlflow.exceptions.MlflowException` if the py:class:`ModelRegistryClient`
                 cannot be created. This may occur, for example, when the registry URI refers
                 to an unsupported store type (e.g., the FileStore).
        :return: A py:class:`ModelRegistryClient` instance
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
                    " '{uri}'. Stores with the following URI schemes are supported:"
                    " {schemes}.".format(uri=self._registry_uri, schemes=exc.supported_uri_schemes),
                    FEATURE_DISABLED,
                )
        return registry_client

    # Tracking API

    def get_run(self, run_id: str) -> Run:
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata -- :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics --
        :py:class:`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the most recently logged value at the largest step for each metric.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
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
            print("run_id: {}".format(run.info.run_id))
            print("params: {}".format(run.data.params))
            print("status: {}".format(run.info.status))

        .. code-block:: text
            :caption: Output

            run_id: e36b42c587a1413ead7c3b6764120618
            params: {'p': '0'}
            status: FINISHED
        """
        return self._tracking_client.get_run(run_id)

    def get_metric_history(self, run_id: str, key: str) -> List[Metric]:
        """
        Return a list of metric objects corresponding to all values logged for a given metric.

        :param run_id: Unique identifier for run
        :param key: Metric name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_metric_info(history):
                for m in history:
                    print("name: {}".format(m.key))
                    print("value: {}".format(m.value))
                    print("step: {}".format(m.step))
                    print("timestamp: {}".format(m.timestamp))
                    print("--")


            # Create a run under the default experiment (whose id is "0"). Since this is low-level
            # CRUD operation, the method will create a run. To end the run, you'll have
            # to explicitly end it.
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            print("run_id: {}".format(run.info.run_id))
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
        tags: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> Run:
        """
        Create a :py:class:`mlflow.entities.Run` object that can be associated with
        metrics, parameters, artifacts, etc.
        Unlike :py:func:`mlflow.projects.run`, creates objects but does not run code.
        Unlike :py:func:`mlflow.start_run`, does not change the "active run" used by
        :py:func:`mlflow.log_param`.

        :param experiment_id: The string ID of the experiment to create a run in.
        :param start_time: If not provided, use the current timestamp.
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`mlflow.entities.RunTag` objects.
        :param run_name: The name of this run.
        :return: :py:class:`mlflow.entities.Run` that was created.

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
            print("Run tags: {}".format(run.data.tags))
            print("Experiment id: {}".format(run.info.experiment_id))
            print("Run id: {}".format(run.info.run_id))
            print("Run name: {}".format(run.info.run_name))
            print("lifecycle_stage: {}".format(run.info.lifecycle_stage))
            print("status: {}".format(run.info.status))

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

    def search_experiments(
        self,
        view_type: int = ViewType.ACTIVE_ONLY,
        max_results: Optional[int] = SEARCH_MAX_RESULTS_DEFAULT,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        page_token=None,
    ) -> PagedList[Experiment]:
        """
        Search for experiments that match the specified search query.

        :param view_type: One of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or ``ALL``
                          defined in :py:class:`mlflow.entities.ViewType`.
        :param max_results: Maximum number of experiments desired. Certain server backend may apply
                            its own limit.
        :param filter_string:
            Filter query string (e.g., ``"name = 'my_experiment'"``), defaults to searching for all
            experiments. The following identifiers, comparators, and logical operators are
            supported.

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

        :param order_by:
            List of columns to order by. The ``order_by`` column can contain an optional ``DESC`` or
            ``ASC`` value (e.g., ``"name DESC"``). The default ordering is ``ASC``, so ``"name"`` is
            equivalent to ``"name ASC"``. If unspecified, defaults to ``["last_update_time DESC"]``,
            which lists experiments updated most recently first. The following fields are supported:

            - ``experiment_id``: Experiment ID
            - ``name``: Experiment name
            - ``creation_time``: Experiment creation time
            - ``last_update_time``: Experiment last update time

        :param page_token: Token specifying the next page of results. It should be obtained from
                           a ``search_experiments`` call.
        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
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
        """
        Retrieve an experiment by experiment_id from the backend store

        :param experiment_id: The experiment ID returned from ``create_experiment``.
        :return: :py:class:`mlflow.entities.Experiment`

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            client = MlflowClient()
            exp_id = client.create_experiment("Experiment")
            experiment = client.get_experiment(exp_id)

            # Show experiment info
            print("Name: {}".format(experiment.name))
            print("Experiment ID: {}".format(experiment.experiment_id))
            print("Artifact Location: {}".format(experiment.artifact_location))
            print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

        .. code-block:: text
            :caption: Output

            Name: Experiment
            Experiment ID: 1
            Artifact Location: file:///.../mlruns/1
            Lifecycle_stage: active
        """
        return self._tracking_client.get_experiment(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """
        Retrieve an experiment by experiment name from the backend store

        :param name: The experiment name, which is case sensitive.
        :return: An instance of :py:class:`mlflow.entities.Experiment`
                 if an experiment with the specified name exists, otherwise None.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Case-sensitive name
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Default")

            # Show experiment info
            print("Name: {}".format(experiment.name))
            print("Experiment ID: {}".format(experiment.experiment_id))
            print("Artifact Location: {}".format(experiment.artifact_location))
            print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

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
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an experiment.

        :param name: The experiment name. Must be unique.
        :param artifact_location: The location to store run artifacts.
                                  If not provided, the server picks an appropriate default.
        :param tags: A dictionary of key-value pairs that are converted into
                                :py:class:`mlflow.entities.ExperimentTag` objects, set as
                                experiment tags upon experiment creation.
        :return: String as an integer ID of the created experiment.

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
            print("Name: {}".format(experiment.name))
            print("Experiment_id: {}".format(experiment.experiment_id))
            print("Artifact Location: {}".format(experiment.artifact_location))
            print("Tags: {}".format(experiment.tags))
            print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

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
        """
        Delete an experiment from the backend store.
        This deletion is a soft-delete, not a permanent deletion.
        Experiment names can not be reused, unless the deleted experiment
        is permanently deleted by a database admin.

        :param experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create an experiment with a name that is unique and case sensitive
            client = MlflowClient()
            experiment_id = client.create_experiment("New Experiment")
            client.delete_experiment(experiment_id)

            # Examine the deleted experiment details.
            experiment = client.get_experiment(experiment_id)
            print("Name: {}".format(experiment.name))
            print("Artifact Location: {}".format(experiment.artifact_location))
            print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

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

        :param experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_experiment_info(experiment):
                print("Name: {}".format(experiment.name))
                print("Experiment Id: {}".format(experiment.experiment_id))
                print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


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

        :param experiment_id: The experiment ID returned from ``create_experiment``.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_experiment_info(experiment):
                print("Name: {}".format(experiment.name))
                print("Experiment_id: {}".format(experiment.experiment_id))
                print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


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
    ) -> None:
        """
        Log a metric against the run ID.

        :param run_id: The run id to which the metric should be logged.
        :param key: Metric name (string). This string may only contain alphanumerics, underscores
                    (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                    All backend stores will support keys up to length 250, but some may
                    support larger keys.
        :param value: Metric value (float). Note that some special values such
                      as +/- Infinity may be replaced by other values depending on the store. For
                      example, the SQLAlchemy store replaces +/- Inf with max / min float values.
                      All backend stores will support values up to length 5000, but some
                      may support larger values.
        :param timestamp: Time when this metric was calculated. Defaults to the current system time.
        :param step: Integer training step (iteration) at which was the metric calculated.
                     Defaults to 0.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(r):
                print("run_id: {}".format(r.info.run_id))
                print("metrics: {}".format(r.data.metrics))
                print("status: {}".format(r.info.status))


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
        self._tracking_client.log_metric(run_id, key, value, timestamp, step)

    def log_param(self, run_id: str, key: str, value: Any) -> Any:
        """
        Log a parameter (e.g. model hyperparameter) against the run ID.

        :param run_id: The run id to which the param should be logged.
        :param key: Parameter name (string). This string may only contain alphanumerics, underscores
                    (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                    All backend stores support keys up to length 250, but some may
                    support larger keys.
        :param value: Parameter value (string, but will be string-ified if not).
                      All backend stores support values up to length 500, but some
                      may support larger values.
        :return: the parameter value that is logged.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(r):
                print("run_id: {}".format(r.info.run_id))
                print("params: {}".format(r.data.params))
                print("status: {}".format(r.info.status))


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
        self._tracking_client.log_param(run_id, key, value)
        return value

    def set_experiment_tag(self, experiment_id: str, key: str, value: Any) -> None:
        """
        Set a tag on the experiment with the specified ID. Value is converted to a string.

        :param experiment_id: String ID of the experiment.
        :param key: Name of the tag.
        :param value: Tag value (converted to a string).

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create an experiment and set its tag
            client = MlflowClient()
            experiment_id = client.create_experiment("Social Media NLP Experiments")
            client.set_experiment_tag(experiment_id, "nlp.framework", "Spark NLP")

            # Fetch experiment metadata information
            experiment = client.get_experiment(experiment_id)
            print("Name: {}".format(experiment.name))
            print("Tags: {}".format(experiment.tags))

        .. code-block:: text
            :caption: Output

            Name: Social Media NLP Experiments
            Tags: {'nlp.framework': 'Spark NLP'}
        """
        self._tracking_client.set_experiment_tag(experiment_id, key, value)

    def set_tag(self, run_id: str, key: str, value: Any) -> None:
        """
        Set a tag on the run with the specified ID. Value is converted to a string.

        :param run_id: String ID of the run.
        :param key: Tag name (string). This string may only contain alphanumerics,
                    underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                    All backend stores will support keys up to length 250, but some may
                    support larger keys.
        :param value: Tag value (string, but will be string-ified if not).
                      All backend stores will support values up to length 5000, but some
                      may support larger values.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print("run_id: {}".format(run.info.run_id))
                print("Tags: {}".format(run.data.tags))


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
        self._tracking_client.set_tag(run_id, key, value)

    def delete_tag(self, run_id: str, key: str) -> None:
        """
        Delete a tag from a run. This is irreversible.

        :param run_id: String ID of the run
        :param key: Name of the tag

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print("run_id: {}".format(run.info.run_id))
                print("Tags: {}".format(run.data.tags))


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
        """
        Update a run with the specified ID to a new status or name.

        :param run_id: The ID of the Run to update.
        :param status: The new status of the run to set, if specified.
                       At least one of ``status`` or ``name`` should be specified.
        :param name: The new name of the run to set, if specified.
                     At least one of ``name`` or ``status`` should be specified.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(run):
                print("run_id: {}".format(run.info.run_id))
                print("run_name: {}".format(run.info.run_name))
                print("status: {}".format(run.info.status))


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
    ) -> None:
        """
        Log multiple metrics, params, and/or tags.

        :param run_id: String ID of the run
        :param metrics: If provided, List of Metric(key, value, timestamp) instances.
        :param params: If provided, List of Param(key, value) instances.
        :param tags: If provided, List of RunTag(key, value) instances.

        Raises an MlflowException if any errors occur.
        :return: None

        .. code-block:: python
            :caption: Example

            import time

            from mlflow import MlflowClient
            from mlflow.entities import Metric, Param, RunTag


            def print_run_info(r):
                print("run_id: {}".format(r.info.run_id))
                print("params: {}".format(r.data.params))
                print("metrics: {}".format(r.data.metrics))
                print("tags: {}".format(r.data.tags))
                print("status: {}".format(r.info.status))


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

            run_id: ef0247fa3205410595acc0f30f620871
            params: {'p': 'p'}
            metrics: {'m': 1.5}
            tags: {'t': 't'}
            status: FINISHED
        """
        self._tracking_client.log_batch(run_id, metrics, params, tags)

    def log_artifact(self, run_id, local_path, artifact_path=None) -> None:
        """
        Write a local file or directory to the remote ``artifact_uri``.

        :param local_path: Path to the file or directory to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            features = "rooms, zipcode, median_price, school_rating, transport"
            with open("features.txt", "w") as f:
                f.write(features)

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)

            # log and fetch the artifact
            client.log_artifact(run.info.run_id, "features.txt")
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print("artifact: {}".format(artifact.path))
                print("is_dir: {}".format(artifact.is_dir))
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: features.txt
            is_dir: False
        """
        self._tracking_client.log_artifact(run_id, local_path, artifact_path)

    def log_artifacts(
        self, run_id: str, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Write a directory of files to the remote ``artifact_uri``.

        :param local_dir: Path to the directory of files to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.

        .. code-block:: python
            :caption: Example

            import os
            import json

            # Create some artifacts data to preserve
            features = "rooms, zipcode, median_price, school_rating, transport"
            data = {"state": "TX", "Available": 25, "Type": "Detached"}

            # Create couple of artifact files under the local directory "data"
            os.makedirs("data", exist_ok=True)
            with open("data/data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            with open("data/features.txt", "w") as f:
                f.write(features)

            # Create a run under the default experiment (whose id is '0'), and log
            # all files in "data" to root artifact_uri/states
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            client.log_artifacts(run.info.run_id, "data", artifact_path="states")
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print("artifact: {}".format(artifact.path))
                print("is_dir: {}".format(artifact.is_dir))
            client.set_terminated(run.info.run_id)

        .. code-block:: text
            :caption: Output

            artifact: states
            is_dir: True
        """
        self._tracking_client.log_artifacts(run_id, local_dir, artifact_path)

    @contextlib.contextmanager
    def _log_artifact_helper(self, run_id, artifact_file):
        """
        Yields a temporary path to store a file, and then calls `log_artifact` against that path.

        :param run_id: String ID of the run.
        :param artifact_file: The run-relative artifact file path in posixpath format.
        :return: Temporary path to store a file.
        """
        norm_path = posixpath.normpath(artifact_file)
        filename = posixpath.basename(norm_path)
        artifact_dir = posixpath.dirname(norm_path)
        artifact_dir = None if artifact_dir == "" else artifact_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, filename)
            yield tmp_path
            self.log_artifact(run_id, tmp_path, artifact_dir)

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        """
        Log text as an artifact.

        :param run_id: String ID of the run.
        :param text: String containing text to log.
        :param artifact_file: The run-relative artifact file path in posixpath format to which
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

    def log_dict(self, run_id: str, dictionary: Any, artifact_file: str) -> None:
        """
        Log a JSON/YAML-serializable object (e.g. `dict`) as an artifact. The serialization
        format (JSON or YAML) is automatically inferred from the extension of `artifact_file`.
        If the file extension doesn't exist or match any of [".json", ".yml", ".yaml"],
        JSON format is used.

        :param run_id: String ID of the run.
        :param dictionary: Dictionary to log.
        :param artifact_file: The run-relative artifact file path in posixpath format to which
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
                    json.dump(dictionary, f, indent=2)

    def log_figure(
        self,
        run_id: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        artifact_file: str,
    ) -> None:
        """
        Log a figure as an artifact. The following figure objects are supported:

        - `matplotlib.figure.Figure`_
        - `plotly.graph_objects.Figure`_

        .. _matplotlib.figure.Figure:
            https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html

        .. _plotly.graph_objects.Figure:
            https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html

        :param run_id: String ID of the run.
        :param figure: Figure to log.
        :param artifact_file: The run-relative artifact file path in posixpath format to which
                              the figure is saved (e.g. "dir/file.png").

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

        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            # `is_matplotlib_figure` is executed only when `matplotlib` is found in `sys.modules`.
            # This allows logging a `plotly` figure in an environment where `matplotlib` is not
            # installed.
            if "matplotlib" in sys.modules and _is_matplotlib_figure(figure):
                figure.savefig(tmp_path)
            elif "plotly" in sys.modules and _is_plotly_figure(figure):
                file_extension = os.path.splitext(artifact_file)[1]
                if file_extension == ".html":
                    figure.write_html(tmp_path, include_plotlyjs="cdn", auto_open=False)
                elif file_extension in [".png", ".jpeg", ".webp", ".svg", ".pdf"]:
                    figure.write_image(tmp_path)
                else:
                    raise TypeError(
                        f"Unsupported file extension for plotly figure: '{file_extension}'"
                    )
            else:
                raise TypeError("Unsupported figure object type: '{}'".format(type(figure)))

    def log_image(
        self, run_id: str, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_file: str
    ) -> None:
        """
        Log an image as an artifact. The following image objects are supported:

        - `numpy.ndarray`_
        - `PIL.Image.Image`_

        .. _numpy.ndarray:
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

        .. _PIL.Image.Image:
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image

        Numpy array support
            - data type (( ) represents a valid value range):

                - bool
                - integer (0 ~ 255)
                - unsigned integer (0 ~ 255)
                - float (0.0 ~ 1.0)

                .. warning::

                    - Out-of-range integer values will be **clipped** to [0, 255].
                    - Out-of-range float values will be **clipped** to [0, 1].

            - shape (H: height, W: width):

                - H x W (Grayscale)
                - H x W x 1 (Grayscale)
                - H x W x 3 (an RGB channel order is assumed)
                - H x W x 4 (an RGBA channel order is assumed)

        :param run_id: String ID of the run.
        :param image: Image to log.
        :param artifact_file: The run-relative artifact file path in posixpath format to which
                              the image is saved (e.g. "dir/image.png").

        .. code-block:: python
            :caption: Numpy Example

            import mlflow
            import numpy as np

            image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

            run = client.create_run(experiment_id="0")
            client.log_image(run.info.run_id, image, "image.png")

        .. code-block:: python
            :caption: Pillow Example

            import mlflow
            from PIL import Image

            image = Image.new("RGB", (100, 100))

            run = client.create_run(experiment_id="0")
            client.log_image(run.info.run_id, image, "image.png")
        """

        def _is_pillow_image(image):
            from PIL.Image import Image

            return isinstance(image, Image)

        def _is_numpy_array(image):
            import numpy as np

            return isinstance(image, np.ndarray)

        def _normalize_to_uint8(x):
            # Based on: https://github.com/matplotlib/matplotlib/blob/06567e021f21be046b6d6dcf00380c1cb9adaf3c/lib/matplotlib/image.py#L684

            is_int = np.issubdtype(x.dtype, np.integer)
            low = 0
            high = 255 if is_int else 1
            if x.min() < low or x.max() > high:
                msg = (
                    "Out-of-range values are detected. "
                    "Clipping array (dtype: '{}') to [{}, {}]".format(x.dtype, low, high)
                )
                _logger.warning(msg)
                x = np.clip(x, low, high)

            # float or bool
            if not is_int:
                x = x * 255

            return x.astype(np.uint8)

        with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
            if "PIL" in sys.modules and _is_pillow_image(image):
                image.save(tmp_path)
            elif "numpy" in sys.modules and _is_numpy_array(image):
                import numpy as np

                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError(
                        "`log_image` requires Pillow to serialize a numpy array as an image. "
                        "Please install it via: pip install Pillow"
                    ) from exc

                # Ref.: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html#numpy-dtype-kind
                valid_data_types = {
                    "b": "bool",
                    "i": "signed integer",
                    "u": "unsigned integer",
                    "f": "floating",
                }

                if image.dtype.kind not in valid_data_types:
                    raise TypeError(
                        f"Invalid array data type: '{image.dtype}'. "
                        f"Must be one of {list(valid_data_types.values())}"
                    )

                if image.ndim not in [2, 3]:
                    raise ValueError(
                        "`image` must be a 2D or 3D array but got a {}D array".format(image.ndim)
                    )

                if (image.ndim == 3) and (image.shape[2] not in [1, 3, 4]):
                    raise ValueError(
                        "Invalid channel length: {}. Must be one of [1, 3, 4]".format(
                            image.shape[2]
                        )
                    )

                # squeeze a 3D grayscale image since `Image.fromarray` doesn't accept it.
                if image.ndim == 3 and image.shape[2] == 1:
                    image = image[:, :, 0]

                image = _normalize_to_uint8(image)

                Image.fromarray(image).save(tmp_path)

            else:
                raise TypeError("Unsupported image object type: '{}'".format(type(image)))

    def _record_logged_model(self, run_id, mlflow_model):
        """
        Record logged model info with the tracking server.

        :param run_id: run_id under which the model has been logged.
        :param mlflow_model: Model info to be recorded.
        """
        self._tracking_client._record_logged_model(run_id, mlflow_model)

    def list_artifacts(self, run_id: str, path=None) -> List[FileInfo]:
        """
        List the artifacts for a run.

        :param run_id: The run to list artifacts from.
        :param path: The run's relative artifact path to list from. By default it is set to None
                     or the root artifact path.
        :return: List of :py:class:`mlflow.entities.FileInfo`

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_artifact_info(artifact):
                print("artifact: {}".format(artifact.path))
                print("is_dir: {}".format(artifact.is_dir))
                print("size: {}".format(artifact.file_size))


            features = "rooms zipcode, median_price, school_rating, transport"
            labels = "price"

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)

            # Create some artifacts and log under the above run
            for file, content in [("features", features), ("labels", labels)]:
                with open("{}.txt".format(file), "w") as f:
                    f.write(content)
                client.log_artifact(run.info.run_id, "{}.txt".format(file))

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

    @deprecated("mlflow.artifacts.download_artifacts", "2.0")
    def download_artifacts(self, run_id: str, path: str, dst_path: Optional[str] = None) -> str:
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        :param run_id: The run to download artifacts from.
        :param path: Relative source path to the desired artifact.
        :param dst_path: Absolute path of the local filesystem destination directory to which to
                         download the specified artifacts. This directory must already exist.
                         If unspecified, the artifacts will either be downloaded to a new
                         uniquely-named directory on the local filesystem or will be returned
                         directly in the case of the LocalArtifactRepository.
        :return: Local path of desired artifact.

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
            print("Artifacts downloaded in: {}".format(local_path))
            print("Artifacts: {}".format(os.listdir(local_path)))

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

        :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
                       Defaults to "FINISHED".
        :param end_time: If not provided, defaults to the current time.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient


            def print_run_info(r):
                print("run_id: {}".format(r.info.run_id))
                print("status: {}".format(r.info.status))


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
            :caption: Output

            run_id: 575fb62af83f469e84806aee24945973
            status: RUNNING
            --
            run_id: 575fb62af83f469e84806aee24945973
            status: KILLED
        """
        self._tracking_client.set_terminated(run_id, status, end_time)

    def delete_run(self, run_id: str) -> None:
        """Deletes a run with the given ID.

        :param run_id: The unique run id to delete.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            print("run_id: {}; lifecycle_stage: {}".format(run_id, run.info.lifecycle_stage))
            print("--")
            client.delete_run(run_id)
            del_run = client.get_run(run_id)
            print("run_id: {}; lifecycle_stage: {}".format(run_id, del_run.info.lifecycle_stage))

        .. code-block:: text
            :caption: Output

            run_id: a61c7a1851324f7094e8d5014c58c8c8; lifecycle_stage: active
            run_id: a61c7a1851324f7094e8d5014c58c8c8; lifecycle_stage: deleted
        """
        self._tracking_client.delete_run(run_id)

    def restore_run(self, run_id: str) -> None:
        """
        Restores a deleted run with the given ID.

        :param run_id: The unique run id to restore.

        .. code-block:: python
            :caption: Example

            from mlflow import MlflowClient

            # Create a run under the default experiment (whose id is '0').
            client = MlflowClient()
            experiment_id = "0"
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            print("run_id: {}; lifecycle_stage: {}".format(run_id, run.info.lifecycle_stage))
            client.delete_run(run_id)
            del_run = client.get_run(run_id)
            print("run_id: {}; lifecycle_stage: {}".format(run_id, del_run.info.lifecycle_stage))
            client.restore_run(run_id)
            rest_run = client.get_run(run_id)
            print("run_id: {}; lifecycle_stage: {}".format(run_id, rest_run.info.lifecycle_stage))

        .. code-block:: text
            :caption: Output

            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: active
            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: deleted
            run_id: 7bc59754d7e74534a7917d62f2873ac0; lifecycle_stage: active
        """
        self._tracking_client.restore_run(run_id)

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        run_view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[Run]:
        """
        Search for Runs that fit the specified criteria.

        :param experiment_ids: List of experiment IDs, or a single int or string id.
        :param filter_string: Filter query string, defaults to searching all runs.
        :param run_view_type: one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL runs
                              defined in :py:class:`mlflow.entities.ViewType`.
        :param max_results: Maximum number of runs desired.
        :param order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
                     can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
                     The default ordering is to sort by ``start_time DESC``, then ``run_id``.
        :param page_token: Token specifying the next page of results. It should be obtained from
            a ``search_runs`` call.

        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
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
                    print("run_id: {}".format(r.info.run_id))
                    print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
                    print("metrics: {}".format(r.data.metrics))

                    # Exclude mlflow system tags
                    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
                    print("tags: {}".format(tags))


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
        self, name: str, tags: Optional[Dict[str, Any]] = None, description: Optional[str] = None
    ) -> RegisteredModel:
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
        :param description: Description of the model.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created by backend.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_model_info(rm):
                print("name: {}".format(rm.name))
                print("tags: {}".format(rm.tags))
                print("description: {}".format(rm.description))


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
        return self._get_registry_client().create_registered_model(name, tags, description)

    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """
        Update registered model name.

        :param name: Name of the registered model to update.
        :param new_name: New proposed name for the registered model.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_model_info(rm):
                print("name: {}".format(rm.name))
                print("tags: {}".format(rm.tags))
                print("description: {}".format(rm.description))


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
        self._get_registry_client().rename_registered_model(name, new_name)

    def update_registered_model(
        self, name: str, description: Optional[str] = None
    ) -> RegisteredModel:
        """
        Updates metadata for RegisteredModel entity. Input field ``description`` should be non-None.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Name of the registered model to update.
        :param description: (Optional) New description.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            def print_registered_model_info(rm):
                print("name: {}".format(rm.name))
                print("tags: {}".format(rm.tags))
                print("description: {}".format(rm.description))


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
        if description is None:
            raise MlflowException("Attempting to update registered model with no new field values.")

        return self._get_registry_client().update_registered_model(
            name=name, description=description
        )

    def delete_registered_model(self, name: str):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Name of the registered model to delete.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_models_info(r_models):
                print("--")
                for rm in r_models:
                    print("name: {}".format(rm.name))
                    print("tags: {}".format(rm.tags))
                    print("description: {}".format(rm.description))


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
        self._get_registry_client().delete_registered_model(name)

    def search_registered_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[RegisteredModel]:
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string
            (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
            defaults to searching for all registered models. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - ``name``: registered model name.
              - ``tags.<tag_key>``: registered model tag. If ``tag_key`` contains spaces, it must be
                wrapped with backticks (e.g., ``"tags.`extra key`"``).

            Comparators
              - ``=``: Equal to.
              - ``!=``: Not equal to.
              - ``LIKE``: Case-sensitive pattern match.
              - ``ILIKE``: Case-insensitive pattern match.

            Logical operators
              - ``AND``: Combines two sub-queries and returns True if both of them are True.

        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient()

            # Get search results filtered by the registered model name
            model_name = "CordobaWeatherForecastModel"
            filter_string = "name='{}'".format(model_name)
            results = client.search_registered_models(filter_string=filter_string)
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

            # Get search results filtered by the registered model name that matches
            # prefix pattern
            filter_string = "name LIKE 'Boston%'"
            results = client.search_registered_models(filter_string=filter_string)
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

            # Get all registered models and order them by ascending order of the names
            results = client.search_registered_models(order_by=["name ASC"])
            print("-" * 80)
            for res in results:
                for mv in res.latest_versions:
                    print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

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
        """
        :param name: Name of the registered model to get.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_model_info(rm):
                print("--")
                print("name: {}".format(rm.name))
                print("tags: {}".format(rm.tags))
                print("description: {}".format(rm.description))


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
        return self._get_registry_client().get_registered_model(name)

    def get_latest_versions(self, name: str, stages: List[str] = None) -> List[ModelVersion]:
        """
        Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        :param name: Name of the registered model from which to get the latest versions.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for ALL_STAGES.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_models_info(mv):
                for m in mv:
                    print("name: {}".format(m.name))
                    print("latest version: {}".format(m.version))
                    print("run_id: {}".format(m.run_id))
                    print("current_stage: {}".format(m.current_stage))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")

            # Create two runs Log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = "runs:/{}/sklearn-model".format(run_id)
                mv = client.create_model_version(name, model_uri, run_id)
                print("model version {} created".format(mv.version))

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
        return self._get_registry_client().get_latest_versions(name, stages)

    def set_registered_model_tag(self, name, key, value) -> None:
        """
        Set a tag for the registered model.

        :param name: Registered model name.
        :param key: Tag key to log.
        :param value: Tag value log.
        :return: None

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
        self._get_registry_client().set_registered_model_tag(name, key, value)

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """
        Delete a tag associated with the registered model.

        :param name: Registered model name.
        :param key: Registered model tag key.
        :return: None

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient


            def print_registered_models_info(r_models):
                print("--")
                for rm in r_models:
                    print("name: {}".format(rm.name))
                    print("tags: {}".format(rm.tags))


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
        self._get_registry_client().delete_registered_model_tag(name, key)

    # Model Version Methods

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    ) -> ModelVersion:
        """
        Create a new model version from given source (artifact URI).

        :param name: Name for the containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
        :param run_link: Link to the run from an MLflow tracking server that generated this model.
        :param description: Description of the version.
        :param await_creation_for: Number of seconds to wait for the model version to finish being
                                    created and is in ``READY`` status. By default, the function
                                    waits for five minutes. Specify 0 or None to skip waiting.
        :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
                 backend.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            desc = "A new version of the model"
            runs_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
            model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
            mv = client.create_model_version(name, model_src, run.info.run_id, description=desc)
            print("Name: {}".format(mv.name))
            print("Version: {}".format(mv.version))
            print("Description: {}".format(mv.description))
            print("Status: {}".format(mv.status))
            print("Stage: {}".format(mv.current_stage))

        .. code-block:: text
            :caption: Output

            Name: RandomForestRegression
            Version: 1
            Description: A new version of the model
            Status: READY
            Stage: None
        """
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
        if is_databricks_uri(self._registry_uri) and tracking_uri != self._registry_uri:
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
                "=== Source model files were copied to %s" % new_source
                + " in the model registry workspace. You may want to delete the files once the"
                + " model version is in 'READY' status. You can also find this location in the"
                + " `source` field of the created model version. ==="
            )
        return self._get_registry_client().create_model_version(
            name=name,
            source=new_source,
            run_id=run_id,
            tags=tags,
            run_link=run_link,
            description=description,
            await_creation_for=await_creation_for,
        )

    def update_model_version(
        self, name: str, version: str, description: Optional[str] = None
    ) -> ModelVersion:
        """
        Update metadata associated with a model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :param description: New description.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Description: {}".format(mv.description))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
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

        return self._get_registry_client().update_model_version(
            name=name, version=version, description=description
        )

    def transition_model_version_stage(
        self, name: str, version: str, stage: str, archive_existing_versions: bool = False
    ) -> ModelVersion:
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set to ``True``, all existing model
            versions in the stage will be automatically moved to the "archived" stage. Only valid
            when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Description: {}".format(mv.description))
                print("Stage: {}".format(mv.current_stage))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            desc = "A new version of the model using ensemble trees"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
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
        return self._get_registry_client().transition_model_version_stage(
            name, version, stage, archive_existing_versions
        )

    def delete_model_version(self, name: str, version: str) -> None:
        """
        Delete model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_models_info(mv):
                for m in mv:
                    print("name: {}".format(m.name))
                    print("latest version: {}".format(m.version))
                    print("run_id: {}".format(m.run_id))
                    print("current_stage: {}".format(m.current_stage))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")

            # Create two runs and log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = "runs:/{}/sklearn-model".format(run_id)
                mv = client.create_model_version(name, model_uri, run_id)
                print("model version {} created".format(mv.version))

            print("--")

            # Fetch latest version; this will be version 2
            models = client.get_latest_versions(name, stages=["None"])
            print_models_info(models)
            print("--")

            # Delete the latest model version 2
            print("Deleting model version {}".format(mv.version))
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
        self._get_registry_client().delete_model_version(name, version)

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """
        :param name: Name of the containing registered model.
        :param version: Version number as an integer of the model version.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor

            # Create two runs Log MLflow entities
            with mlflow.start_run() as run1:
                params = {"n_estimators": 3, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            with mlflow.start_run() as run2:
                params = {"n_estimators": 6, "random_state": 42}
                rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            name = "RandomForestRegression"
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a two versions of the rfr model under the registered model name
            for run_id in [run1.info.run_id, run2.info.run_id]:
                model_uri = "runs:/{}/sklearn-model".format(run_id)
                mv = client.create_model_version(name, model_uri, run_id)
                print("model version {} created".format(mv.version))
            print("--")

            # Fetch the last version; this will be version 2
            mv = client.get_model_version(name, mv.version)
            print_model_version_info(mv)

        .. code-block:: text
            :caption: Output

            model version 1 created
            model version 2 created
            --
            Name: RandomForestRegression
            Version: 2
        """
        return self._get_registry_client().get_model_version(name, version)

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """
        Get the download location in Model Registry for this model version.

        :param name: Name of the containing registered model.
        :param version: Version number as an integer of the model version.
        :return: A single URI location that allows reads for downloading.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="models/sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            model_uri = "runs:/{}/models/sklearn-model".format(run.info.run_id)
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            artifact_uri = client.get_model_version_download_uri(name, mv.version)
            print("Download URI: {}".format(artifact_uri))

        .. code-block:: text
            :caption: Output

            Download URI: runs:/44e04097ac364cd895f2039eaccca9ac/models/sklearn-model
        """
        return self._get_registry_client().get_model_version_download_uri(name, version)

    def search_model_versions(
        self,
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[ModelVersion]:
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: Filter query string
            (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
            defaults to searching for all model versions. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - ``name``: model name.
              - ``source_path``: model version source path.
              - ``run_id``: The id of the mlflow run that generates the model version.
              - ``tags.<tag_key>``: model version tag. If ``tag_key`` contains spaces, it must be
                wrapped with backticks (e.g., ``"tags.`extra key`"``).

            Comparators
              - ``=``: Equal to.
              - ``!=``: Not equal to.
              - ``LIKE``: Case-sensitive pattern match.
              - ``ILIKE``: Case-insensitive pattern match.
              - ``IN``: In a value list. Only ``run_id`` identifier supports ``IN`` comparator.

            Logical operators
              - ``AND``: Combines two sub-queries and returns True if both of them are True.

        :param max_results: Maximum number of model versions desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_model_versions`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects that satisfy the search expressions. The pagination token for the next
                 page can be obtained via the ``token`` attribute of the object.

        .. code-block:: python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient()

            # Get all versions of the model filtered by name
            model_name = "CordobaWeatherForecastModel"
            filter_string = "name='{}'".format(model_name)
            results = client.search_model_versions(filter_string)
            print("-" * 80)
            for res in results:
                print("name={}; run_id={}; version={}".format(res.name, res.run_id, res.version))

            # Get the version of the model filtered by run_id
            run_id = "e14afa2f47a040728060c1699968fd43"
            filter_string = "run_id='{}'".format(run_id)
            results = client.search_model_versions(filter_string)
            print("-" * 80)
            for res in results:
                print("name={}; run_id={}; version={}".format(res.name, res.run_id, res.version))

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

    def get_model_version_stages(
        self, name: str, version: str  # pylint: disable=unused-argument
    ) -> List[str]:
        """
        :return: A list of valid stages.

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="models/sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # fetch valid stages
            model_uri = "runs:/{}/models/sklearn-model".format(run.info.run_id)
            mv = client.create_model_version(name, model_uri, run.info.run_id)
            stages = client.get_model_version_stages(name, mv.version)
            print("Model list of valid stages: {}".format(stages))

        .. code-block:: text
            :caption: Output

            Model list of valid stages: ['None', 'Staging', 'Production', 'Archived']
        """
        return ALL_STAGES

    def set_model_version_tag(
        self, name: str, version: str = None, key: str = None, value: Any = None, stage: str = None
    ) -> None:
        """
        Set a tag for the model version.
        When stage is set, tag will be set for latest model version of the stage.
        Setting both version and stage parameter will result in error.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key to log. key is required.
        :param value: Tag value to log. value is required.
        :param stage: Registered model stage.
        :return: None

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Tags: {}".format(mv.tags))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # and set a tag
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
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
        if stage:
            latest_versions = self.get_latest_versions(name, stages=[stage])
            if not latest_versions:
                raise MlflowException(f"Could not find any model version for {stage} stage")
            version = latest_versions[0].version

        self._get_registry_client().set_model_version_tag(name, version, key, value)

    def delete_model_version_tag(
        self, name: str, version: str = None, key: str = None, stage: str = None
    ) -> None:
        """
        Delete a tag associated with the model version.
        When stage is set, tag will be deleted for latest model version of the stage.
        Setting both version and stage parameter will result in error.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key. key is required.
        :param stage: Registered model stage.
        :return: None

        .. code-block:: python
            :caption: Example

            import mlflow.sklearn
            from mlflow import MlflowClient
            from sklearn.ensemble import RandomForestRegressor


            def print_model_version_info(mv):
                print("Name: {}".format(mv.name))
                print("Version: {}".format(mv.version))
                print("Tags: {}".format(mv.tags))


            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            params = {"n_estimators": 3, "random_state": 42}
            name = "RandomForestRegression"
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

            # Register model name in the model registry
            client = MlflowClient()
            client.create_registered_model(name)

            # Create a new version of the rfr model under the registered model name
            # and delete a tag
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
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
        if stage:
            latest_versions = self.get_latest_versions(name, stages=[stage])
            if not latest_versions:
                raise MlflowException("Could not find any model version for {stage} stage")
            version = latest_versions[0].version
        self._get_registry_client().delete_model_version_tag(name, version, key)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set a registered model alias pointing to a model version.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :param version: Registered model version number.
        :return: None

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
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
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

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
        self._get_registry_client().set_registered_model_alias(name, alias, version)

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        """
        Delete an alias associated with a registered model.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: None

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
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
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

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
            client.set_registered_model_alias(name, "test-alias")
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
        self._get_registry_client().delete_registered_model_alias(name, alias)

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        """
        Get the model version instance by name and alias.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        .. code-block:: Python
            :caption: Example

            import mlflow
            from mlflow import MlflowClient
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
            rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])

            # Log MLflow entities
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

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
        _validate_model_alias_name(alias)
        return self._get_registry_client().get_model_version_by_alias(name, alias)
