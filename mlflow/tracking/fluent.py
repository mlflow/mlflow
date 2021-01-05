"""
Internal module implementing the fluent API, allowing management of an active
MLflow run. This module is exposed to users at the top-level :py:mod:`mlflow` module.
"""
import os

import atexit
import time
import logging
import inspect
import numpy as np
import pandas as pd

from mlflow.entities import Run, RunStatus, Param, RunTag, Metric, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking import artifact_utils, _get_store
from mlflow.tracking.context import registry as context_registry
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils import env
from mlflow.utils.autologging_utils import _is_testing, autologging_integration
from mlflow.utils.databricks_utils import is_in_databricks_notebook, get_notebook_id
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.validation import _validate_run_id
from mlflow.utils.annotations import experimental

from mlflow import (
    tensorflow,
    keras,
    gluon,
    xgboost,
    lightgbm,
    statsmodels,
    spark,
    sklearn,
    fastai,
    pytorch,
)

_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_EXPERIMENT_NAME_ENV_VAR = "MLFLOW_EXPERIMENT_NAME"
_RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
_active_run_stack = []
_active_experiment_id = None

SEARCH_MAX_RESULTS_PANDAS = 100000
NUM_RUNS_PER_PAGE_PANDAS = 10000

_logger = logging.getLogger(__name__)


def set_experiment(experiment_name):
    """
    Set given experiment as active experiment. If experiment does not exist, create an experiment
    with provided name.

    :param experiment_name: Case sensitive name of an experiment to be activated.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Set an experiment name, which must be unique and case sensitive.
        mlflow.set_experiment("Social NLP Experiments")

        # Get Experiment Details
        experiment = mlflow.get_experiment_by_name("Social NLP Experiments")
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        Experiment_id: 1
        Artifact Location: file:///.../mlruns/1
        Tags: {}
        Lifecycle_stage: active
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    exp_id = experiment.experiment_id if experiment else None
    if exp_id is None:  # id can be 0
        print("INFO: '{}' does not exist. Creating a new experiment".format(experiment_name))
        exp_id = client.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == LifecycleStage.DELETED:
        raise MlflowException(
            "Cannot set a deleted experiment '%s' as the active experiment."
            " You can restore the experiment, or permanently delete the "
            " experiment to create a new one." % experiment.name
        )
    global _active_experiment_id
    _active_experiment_id = exp_id


class ActiveRun(Run):  # pylint: disable=W0223
    """Wrapper around :py:class:`mlflow.entities.Run` to enable using Python ``with`` syntax."""

    def __init__(self, run):
        Run.__init__(self, run.info, run.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED
        end_run(RunStatus.to_string(status))
        return exc_type is None


def start_run(run_id=None, experiment_id=None, run_name=None, nested=False, tags=None):
    """
    Start a new MLflow run, setting it as the active run under which metrics and parameters
    will be logged. The return value can be used as a context manager within a ``with`` block;
    otherwise, you must call ``end_run()`` to terminate the current run.

    If you pass a ``run_id`` or the ``MLFLOW_RUN_ID`` environment variable is set,
    ``start_run`` attempts to resume a run with the specified run ID and
    other parameters are ignored. ``run_id`` takes precedence over ``MLFLOW_RUN_ID``.

    If resuming an existing run, the run status is set to ``RunStatus.RUNNING``.

    MLflow sets a variety of default tags on the run, as defined in
    :ref:`MLflow system tags <system_tags>`.

    :param run_id: If specified, get the run with the specified UUID and log parameters
                     and metrics under that run. The run's end time is unset and its status
                     is set to running, but the run's other attributes (``source_version``,
                     ``source_type``, etc.) are not changed.
    :param experiment_id: ID of the experiment under which to create the current run (applicable
                          only when ``run_id`` is not specified). If ``experiment_id`` argument
                          is unspecified, will look for valid experiment in the following order:
                          activated using ``set_experiment``, ``MLFLOW_EXPERIMENT_NAME``
                          environment variable, ``MLFLOW_EXPERIMENT_ID`` environment variable,
                          or the default experiment as defined by the tracking server.
    :param run_name: Name of new run (stored as a ``mlflow.runName`` tag).
                     Used only when ``run_id`` is unspecified.
    :param nested: Controls whether run is nested in parent run. ``True`` creates a nested run.
    :param tags: An optional dictionary of string keys and values to set as tags on the new run.
    :return: :py:class:`mlflow.ActiveRun` object that acts as a context manager wrapping
             the run's state.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Create nested runs
        with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
            mlflow.log_param("parent", "yes")
            with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
                mlflow.log_param("child", "yes")

        print("parent run_id: {}".format(parent_run.info.run_id))
        print("child run_id : {}".format(child_run.info.run_id))
        print("--")

        # Search all child runs with a parent id
        query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
        results = mlflow.search_runs(filter_string=query)
        print(results[["run_id", "params.child", "tags.mlflow.runName"]])

    .. code-block:: text
        :caption: Output

        parent run_id: 5ec0e7ae18f54c2694ffb48c2fccf25c
        child run_id : 78b3b0d264b44cd29e8dc389749bb4be
        --
                                     run_id params.child tags.mlflow.runName
        0  78b3b0d264b44cd29e8dc389749bb4be          yes           CHILD_RUN
    """
    global _active_run_stack
    # back compat for int experiment_id
    experiment_id = str(experiment_id) if isinstance(experiment_id, int) else experiment_id
    if len(_active_run_stack) > 0 and not nested:
        raise Exception(
            (
                "Run with UUID {} is already active. To start a new run, first end the "
                + "current run with mlflow.end_run(). To start a nested "
                + "run, call start_run with nested=True"
            ).format(_active_run_stack[0].info.run_id)
        )
    if run_id:
        existing_run_id = run_id
    elif _RUN_ID_ENV_VAR in os.environ:
        existing_run_id = os.environ[_RUN_ID_ENV_VAR]
        del os.environ[_RUN_ID_ENV_VAR]
    else:
        existing_run_id = None
    if existing_run_id:
        _validate_run_id(existing_run_id)
        active_run_obj = MlflowClient().get_run(existing_run_id)
        # Check to see if experiment_id from environment matches experiment_id from set_experiment()
        if (
            _active_experiment_id is not None
            and _active_experiment_id != active_run_obj.info.experiment_id
        ):
            raise MlflowException(
                "Cannot start run with ID {} because active run ID "
                "does not match environment run ID. Make sure --experiment-name "
                "or --experiment-id matches experiment set with "
                "set_experiment(), or just use command-line "
                "arguments".format(existing_run_id)
            )
        # Check to see if current run isn't deleted
        if active_run_obj.info.lifecycle_stage == LifecycleStage.DELETED:
            raise MlflowException(
                "Cannot start run with ID {} because it is in the "
                "deleted state.".format(existing_run_id)
            )
        # Use previous end_time because a value is required for update_run_info
        end_time = active_run_obj.info.end_time
        _get_store().update_run_info(
            existing_run_id, run_status=RunStatus.RUNNING, end_time=end_time
        )
        active_run_obj = MlflowClient().get_run(existing_run_id)
    else:
        if len(_active_run_stack) > 0:
            parent_run_id = _active_run_stack[-1].info.run_id
        else:
            parent_run_id = None

        exp_id_for_run = experiment_id if experiment_id is not None else _get_experiment_id()

        user_specified_tags = tags or {}
        if parent_run_id is not None:
            user_specified_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        if run_name is not None:
            user_specified_tags[MLFLOW_RUN_NAME] = run_name

        tags = context_registry.resolve_tags(user_specified_tags)

        active_run_obj = MlflowClient().create_run(experiment_id=exp_id_for_run, tags=tags)

    _active_run_stack.append(ActiveRun(active_run_obj))
    return _active_run_stack[-1]


def end_run(status=RunStatus.to_string(RunStatus.FINISHED)):
    """End an active MLflow run (if there is one).

    .. code-block:: python
        :caption: Example

        import mlflow

        # Start run and get status
        mlflow.start_run()
        run = mlflow.active_run()
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))

        # End run and get status
        mlflow.end_run()
        run = mlflow.get_run(run.info.run_id)
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        print("--")

        # Check for any active runs
        print("Active run: {}".format(mlflow.active_run()))

    .. code-block:: text
        :caption: Output

        run_id: b47ee4563368419880b44ad8535f6371; status: RUNNING
        run_id: b47ee4563368419880b44ad8535f6371; status: FINISHED
        --
        Active run: None
    """
    global _active_run_stack
    if len(_active_run_stack) > 0:
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_ID_ENV_VAR)
        run = _active_run_stack.pop()
        MlflowClient().set_terminated(run.info.run_id, status)


atexit.register(end_run)


def active_run():
    """Get the currently active ``Run``, or None if no such run exists.

    **Note**: You cannot access currently-active run attributes
    (parameters, metrics, etc.) through the run returned by ``mlflow.active_run``. In order
    to access such attributes, use the :py:class:`mlflow.tracking.MlflowClient` as follows:

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.start_run()
        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))
        mlflow.end_run()

    .. code-block:: text
        :caption: Output

        Active run_id: 6f252757005748708cd3aad75d1ff462
    """
    return _active_run_stack[-1] if len(_active_run_stack) > 0 else None


def get_run(run_id):
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

        with mlflow.start_run() as run:
            mlflow.log_param("p", 0)

        run_id = run.info.run_id
        print("run_id: {}; lifecycle_stage: {}".format(run_id,
            mlflow.get_run(run_id).info.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        run_id: 7472befefc754e388e8e922824a0cca5; lifecycle_stage: active
    """
    return MlflowClient().get_run(run_id)


def log_param(key, value):
    """
    Log a parameter under the current run. If no run is active, this method will create
    a new active run.

    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-ified if not)

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
            mlflow.log_param("learning_rate", 0.01)
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_param(run_id, key, value)


def set_tag(key, value):
    """
    Set a tag under the current run. If no run is active, this method will create a
    new active run.

    :param key: Tag name (string)
    :param value: Tag value (string, but will be string-ified if not)

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
           mlflow.set_tag("release.version", "2.2.0")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().set_tag(run_id, key, value)


def delete_tag(key):
    """
    Delete a tag from a run. This is irreversible. If no run is active, this method
    will create a new active run.

    :param key: Name of the tag

    .. code-block:: python
        :caption: Example

        import mlflow

        tags = {"engineering": "ML Platform",
                "engineering_remote": "ML Platform"}

        with mlflow.start_run() as run:
            mlflow.set_tags(tags)

        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.delete_tag("engineering_remote")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().delete_tag(run_id, key)


def log_metric(key, value, step=None):
    """
    Log a metric under the current run. If no run is active, this method will create
    a new active run.

    :param key: Metric name (string).
    :param value: Metric value (float). Note that some special values such as +/- Infinity may be
                  replaced by other values depending on the store. For example, the
                  SQLAlchemy store replaces +/- Infinity with max / min float values.
    :param step: Metric step (int). Defaults to zero if unspecified.

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
            mlflow.log_metric("mse", 2500.00)
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_metric(run_id, key, value, int(time.time() * 1000), step or 0)


def log_metrics(metrics, step=None):
    """
    Log multiple metrics for the current run. If no run is active, this method will create a new
    active run.

    :param metrics: Dictionary of metric_name: String -> value: Float. Note that some special
                    values such as +/- Infinity may be replaced by other values depending on
                    the store. For example, sql based store may replace +/- Infinity with
                    max / min float values.
    :param step: A single integer step at which to log the specified
                 Metrics. If unspecified, each metric is logged at step zero.

    :returns: None

    .. code-block:: python
        :caption: Example

        import mlflow

        metrics = {"mse": 2500.00, "rmse": 50.00}

        # Log a batch of metrics
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
    """
    run_id = _get_or_start_run().info.run_id
    timestamp = int(time.time() * 1000)
    metrics_arr = [Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=[])


def log_params(params):
    """
    Log a batch of params for the current run. If no run is active, this method will create a
    new active run.

    :param params: Dictionary of param_name: String -> value: (String, but will be string-ified if
                   not)
    :returns: None

    .. code-block:: python
        :caption: Example

        import mlflow

        params = {"learning_rate": 0.01, "n_estimators": 10}

        # Log a batch of parameters
        with mlflow.start_run():
            mlflow.log_params(params)
    """
    run_id = _get_or_start_run().info.run_id
    params_arr = [Param(key, str(value)) for key, value in params.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=params_arr, tags=[])


def set_tags(tags):
    """
    Log a batch of tags for the current run. If no run is active, this method will create a
    new active run.

    :param tags: Dictionary of tag_name: String -> value: (String, but will be string-ified if
                 not)
    :returns: None

    .. code-block:: python
        :caption: Example

        import mlflow

        tags = {"engineering": "ML Platform",
                "release.candidate": "RC1",
                "release.version": "2.2.0"}

        # Set a batch of tags
        with mlflow.start_run():
            mlflow.set_tags(tags)
    """
    run_id = _get_or_start_run().info.run_id
    tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=[], tags=tags_arr)


def log_artifact(local_path, artifact_path=None):
    """
    Log a local file or directory as an artifact of the currently active run. If no run is
    active, this method will create a new active run.

    :param local_path: Path to the file to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Create a features.txt artifact file
        features = "rooms, zipcode, median_price, school_rating, transport"
        with open("features.txt", 'w') as f:
            f.write(features)

        # With artifact_path=None write features.txt under
        # root artifact_uri/artifacts directory
        with mlflow.start_run():
            mlflow.log_artifact("features.txt")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_artifact(run_id, local_path, artifact_path)


def log_artifacts(local_dir, artifact_path=None):
    """
    Log all the contents of a local directory as artifacts of the run. If no run is active,
    this method will create a new active run.

    :param local_dir: Path to the directory of files to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.

    .. code-block:: python
        :caption: Example

        import os
        import mlflow

        # Create some files to preserve as artifacts
        features = "rooms, zipcode, median_price, school_rating, transport"
        data = {"state": "TX", "Available": 25, "Type": "Detached"}

        # Create couple of artifact files under the directory "data"
        os.makedirs("data", exist_ok=True)
        with open("data/data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        with open("data/features.txt", 'w') as f:
            f.write(features)

        # Write all files in "data" to root artifact_uri/states
        with mlflow.start_run():
            mlflow.log_artifacts("data", artifact_path="states")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_artifacts(run_id, local_dir, artifact_path)


def log_text(text, artifact_file):
    """
    Log text as an artifact.

    :param text: String containing text to log.
    :param artifact_file: The run-relative artifact file path in posixpath format to which
                          the text is saved (e.g. "dir/file.txt").

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
            # Log text to a file under the run's root artifact directory
            mlflow.log_text("text1", "file1.txt")

            # Log text in a subdirectory of the run's root artifact directory
            mlflow.log_text("text2", "dir/file2.txt")

            # Log HTML text
            mlflow.log_text("<h1>header</h1>", "index.html")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_text(run_id, text, artifact_file)


@experimental
def log_dict(dictionary, artifact_file):
    """
    Log a dictionary as an artifact. The serialization format (JSON or YAML) is automatically
    inferred from the extension of `artifact_file`. If the file extension doesn't exist or match
    any of [".json", ".yml", ".yaml"], JSON format is used.

    :param dictionary: Dictionary to log.
    :param artifact_file: The run-relative artifact file path in posixpath format to which
                            the dictionary is saved (e.g. "dir/data.json").

    .. code-block:: python
        :caption: Example

        import mlflow

        dictionary = {"k": "v"}

        with mlflow.start_run():
            # Log a dictionary as a JSON file under the run's root artifact directory
            mlflow.log_dict(dictionary, "data.json")

            # Log a dictionary as a YAML file in a subdirectory of the run's root artifact directory
            mlflow.log_dict(dictionary, "dir/data.yml")

            # If the file extension doesn't exist or match any of [".json", ".yaml", ".yml"],
            # JSON format is used.
            mlflow.log_dict(dictionary, "data")
            mlflow.log_dict(dictionary, "data.txt")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_dict(run_id, dictionary, artifact_file)


@experimental
def log_figure(figure, artifact_file):
    """
    Log a figure as an artifact. The following figure objects are supported:

    - `matplotlib.figure.Figure`_
    - `plotly.graph_objects.Figure`_

    .. _matplotlib.figure.Figure:
        https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html

    .. _plotly.graph_objects.Figure:
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html

    :param figure: Figure to log.
    :param artifact_file: The run-relative artifact file path in posixpath format to which
                          the figure is saved (e.g. "dir/file.png").

    .. code-block:: python
        :caption: Matplotlib Example

        import mlflow
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [2, 3])

        with mlflow.start_run():
            mlflow.log_figure(fig, "figure.png")

    .. code-block:: python
        :caption: Plotly Example

        import mlflow
        from plotly import graph_objects as go

        fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

        with mlflow.start_run():
            mlflow.log_figure(fig, "figure.html")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_figure(run_id, figure, artifact_file)


@experimental
def log_image(image, artifact_file):
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

    :param image: Image to log.
    :param artifact_file: The run-relative artifact file path in posixpath format to which
                          the image is saved (e.g. "dir/image.png").

    .. code-block:: python
        :caption: Numpy Example

        import mlflow
        import numpy as np

        image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

        with mlflow.start_run():
            mlflow.log_image(image, "image.png")

    .. code-block:: python
        :caption: Pillow Example

        import mlflow
        from PIL import Image

        image = Image.new("RGB", (100, 100))

        with mlflow.start_run():
            mlflow.log_image(image, "image.png")
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_image(run_id, image, artifact_file)


def _record_logged_model(mlflow_model):
    run_id = _get_or_start_run().info.run_id
    MlflowClient()._record_logged_model(run_id, mlflow_model)


def get_experiment(experiment_id):
    """
    Retrieve an experiment by experiment_id from the backend store

    :param experiment_id: The string-ified experiment ID returned from ``create_experiment``.
    :return: :py:class:`mlflow.entities.Experiment`

    .. code-block:: python
        :caption: Example

        import mlflow

        experiment = mlflow.get_experiment("0")
        print("Name: {}".format(experiment.name))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        Name: Default
        Artifact Location: file:///.../mlruns/0
        Tags: {}
        Lifecycle_stage: active
    """
    return MlflowClient().get_experiment(experiment_id)


def get_experiment_by_name(name):
    """
    Retrieve an experiment by experiment name from the backend store

    :param name: The case senstive experiment name.
    :return: :py:class:`mlflow.entities.Experiment`

    .. code-block:: python
        :caption: Example

        import mlflow

        # Case sensitive name
        experiment = mlflow.get_experiment_by_name("Default")
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        Experiment_id: 0
        Artifact Location: file:///.../mlruns/0
        Tags: {}
        Lifecycle_stage: active
    """
    return MlflowClient().get_experiment_by_name(name)


def create_experiment(name, artifact_location=None):
    """
    Create an experiment.

    :param name: The experiment name, which must be unique and is case sensitive
    :param artifact_location: The location to store run artifacts.
                              If not provided, the server picks an appropriate default.
    :return: String ID of the created experiment.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Create an experiment name, which must be unique and case sensitive
        experiment_id = mlflow.create_experiment("Social NLP Experiments")
        experiment = mlflow.get_experiment(experiment_id)
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        Name: Social NLP Experiments
        Experiment_id: 1
        Artifact Location: file:///.../mlruns/1
        Tags= {}
        Lifecycle_stage: active
    """
    return MlflowClient().create_experiment(name, artifact_location)


def delete_experiment(experiment_id):
    """
    Delete an experiment from the backend store.

    :param experiment_id: The The string-ified experiment ID returned from ``create_experiment``.

    .. code-block:: python
        :caption: Example

        import mlflow

        experiment_id = mlflow.create_experiment("New Experiment")
        mlflow.delete_experiment(experiment_id)

        # Examine the deleted experiment details.
        experiment = mlflow.get_experiment(experiment_id)
        print("Name: {}".format(experiment.name))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        Name: New Experiment
        Artifact Location: file:///.../mlruns/2
        Lifecycle_stage: deleted
    """
    MlflowClient().delete_experiment(experiment_id)


def delete_run(run_id):
    """
    Deletes a run with the given ID.

    :param run_id: Unique identifier for the run to delete.

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run() as run:
            mlflow.log_param("p", 0)

        run_id = run.info.run_id
        mlflow.delete_run(run_id)

        print("run_id: {}; lifecycle_stage: {}".format(run_id,
            mlflow.get_run(run_id).info.lifecycle_stage))

    .. code-block:: text
        :caption: Output

        run_id: 45f4af3e6fd349e58579b27fcb0b8277; lifecycle_stage: deleted
    """
    MlflowClient().delete_run(run_id)


def get_artifact_uri(artifact_path=None):
    """
    Get the absolute URI of the specified artifact in the currently active run.
    If `path` is not specified, the artifact root URI of the currently active
    run will be returned; calls to ``log_artifact`` and ``log_artifacts`` write
    artifact(s) to subdirectories of the artifact root URI.

    If no run is active, this method will create a new active run.

    :param artifact_path: The run-relative artifact path for which to obtain an absolute URI.
                          For example, "path/to/artifact". If unspecified, the artifact root URI
                          for the currently active run will be returned.
    :return: An *absolute* URI referring to the specified artifact or the currently adtive run's
             artifact root. For example, if an artifact path is provided and the currently active
             run uses an S3-backed store, this may be a uri of the form
             ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
             is not provided and the currently active run uses an S3-backed store, this may be a
             URI of the form ``s3://<bucket_name>/path/to/artifact/root``.

    .. code-block:: python
        :caption: Example

        import mlflow

        features = "rooms, zipcode, median_price, school_rating, transport"
        with open("features.txt", 'w') as f:
            f.write(features)

        # Log the artifact in a directory "features" under the root artifact_uri/features
        with mlflow.start_run():
            mlflow.log_artifact("features.txt", artifact_path="features")

            # Fetch the artifact uri root directory
            artifact_uri = mlflow.get_artifact_uri()
            print("Artifact uri: {}".format(artifact_uri))

            # Fetch a specific artifact uri
            artifact_uri = mlflow.get_artifact_uri(artifact_path="features/features.txt")
            print("Artifact uri: {}".format(artifact_uri))

    .. code-block:: text
        :caption: Output

        Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts
        Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts/features/features.txt
    """
    return artifact_utils.get_artifact_uri(
        run_id=_get_or_start_run().info.run_id, artifact_path=artifact_path
    )


def search_runs(
    experiment_ids=None,
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=SEARCH_MAX_RESULTS_PANDAS,
    order_by=None,
):
    """
    Get a pandas DataFrame of runs that fit the search criteria.

    :param experiment_ids: List of experiment IDs. None will default to the active experiment.
    :param filter_string: Filter query string, defaults to searching all runs.
    :param run_view_type: one of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or ``ALL`` runs
                            defined in :py:class:`mlflow.entities.ViewType`.
    :param max_results: The maximum number of runs to put in the dataframe. Default is 100,000
                        to avoid causing out-of-memory issues on the user's machine.
    :param order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
                     can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
                     The default ordering is to sort by ``start_time DESC``, then ``run_id``.

    :return: A pandas.DataFrame of runs, where each metric, parameter, and tag
        are expanded into their own columns named metrics.*, params.*, and tags.*
        respectively. For runs that don't have a particular metric, parameter, or tag, their
        value will be (NumPy) Nan, None, or None respectively.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Create an experiment and log two runs under it
        experiment_id = mlflow.create_experiment("Social NLP Experiments")
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metric("m", 1.55)
            mlflow.set_tag("s.release", "1.1.0-RC")
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metric("m", 2.50)
            mlflow.set_tag("s.release", "1.2.0-GA")

        # Search all runs in experiment_id
        df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
        print(df[["metrics.m", "tags.s.release", "run_id"]])
        print("--")

        # Search the experiment_id using a filter_string with tag
        # that has a case insensitive pattern
        filter_string = "tags.s.release ILIKE '%rc%'"
        df = mlflow.search_runs([experiment_id], filter_string=filter_string)
        print(df[["metrics.m", "tags.s.release", "run_id"]])

    .. code-block:: text
        :caption: Output

           metrics.m tags.s.release                            run_id
        0       2.50       1.2.0-GA  147eed886ab44633902cc8e19b2267e2
        1       1.55       1.1.0-RC  5cc7feaf532f496f885ad7750809c4d4
        --
           metrics.m tags.s.release                            run_id
        0       1.55       1.1.0-RC  5cc7feaf532f496f885ad7750809c4d4
    """
    if not experiment_ids:
        experiment_ids = _get_experiment_id()

    # Using an internal function as the linter doesn't like assigning a lambda, and inlining the
    # full thing is a mess
    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_runs(
            experiment_ids, filter_string, run_view_type, number_to_get, order_by, next_page_token
        )

    runs = _paginate(pagination_wrapper_func, NUM_RUNS_PER_PAGE_PANDAS, max_results)

    info = {
        "run_id": [],
        "experiment_id": [],
        "status": [],
        "artifact_uri": [],
        "start_time": [],
        "end_time": [],
    }
    params, metrics, tags = ({}, {}, {})
    PARAM_NULL, METRIC_NULL, TAG_NULL = (None, np.nan, None)
    for i, run in enumerate(runs):
        info["run_id"].append(run.info.run_id)
        info["experiment_id"].append(run.info.experiment_id)
        info["status"].append(run.info.status)
        info["artifact_uri"].append(run.info.artifact_uri)
        info["start_time"].append(pd.to_datetime(run.info.start_time, unit="ms", utc=True))
        info["end_time"].append(pd.to_datetime(run.info.end_time, unit="ms", utc=True))

        # Params
        param_keys = set(params.keys())
        for key in param_keys:
            if key in run.data.params:
                params[key].append(run.data.params[key])
            else:
                params[key].append(PARAM_NULL)
        new_params = set(run.data.params.keys()) - param_keys
        for p in new_params:
            params[p] = [PARAM_NULL] * i  # Fill in null values for all previous runs
            params[p].append(run.data.params[p])

        # Metrics
        metric_keys = set(metrics.keys())
        for key in metric_keys:
            if key in run.data.metrics:
                metrics[key].append(run.data.metrics[key])
            else:
                metrics[key].append(METRIC_NULL)
        new_metrics = set(run.data.metrics.keys()) - metric_keys
        for m in new_metrics:
            metrics[m] = [METRIC_NULL] * i
            metrics[m].append(run.data.metrics[m])

        # Tags
        tag_keys = set(tags.keys())
        for key in tag_keys:
            if key in run.data.tags:
                tags[key].append(run.data.tags[key])
            else:
                tags[key].append(TAG_NULL)
        new_tags = set(run.data.tags.keys()) - tag_keys
        for t in new_tags:
            tags[t] = [TAG_NULL] * i
            tags[t].append(run.data.tags[t])

    data = {}
    data.update(info)
    for key in metrics:
        data["metrics." + key] = metrics[key]
    for key in params:
        data["params." + key] = params[key]
    for key in tags:
        data["tags." + key] = tags[key]
    return pd.DataFrame(data)


def list_run_infos(
    experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=SEARCH_MAX_RESULTS_DEFAULT,
    order_by=None,
):
    """
    Return run information for runs which belong to the experiment_id.

    :param experiment_id: The experiment id which to search
    :param run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs
    :param max_results: Maximum number of results desired.
    :param order_by: List of order_by clauses. Currently supported values are
           are ``metric.key``, ``parameter.key``, ``tag.key``, ``attribute.key``.
           For example, ``order_by=["tag.release ASC", "metric.click_rate DESC"]``.

    :return: A list of :py:class:`mlflow.entities.RunInfo` objects that satisfy the
        search expressions.

    .. code-block:: python
        :caption: Example

        import mlflow
        from mlflow.entities import ViewType

        # Create two runs
        with mlflow.start_run() as run1:
            mlflow.log_param("p", 0)

        with mlflow.start_run() as run2:
            mlflow.log_param("p", 1)

        # Delete the last run
        mlflow.delete_run(run2.info.run_id)

        def print_run_infos(run_infos):
            for r in run_infos:
                print("- run_id: {}, lifecycle_stage: {}".format(r.run_id, r.lifecycle_stage))

        print("Active runs:")
        print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ACTIVE_ONLY))

        print("Deleted runs:")
        print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.DELETED_ONLY))

        print("All runs:")
        print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ALL))

    .. code-block:: text
        :caption: Output

        Active runs:
        - run_id: 4937823b730640d5bed9e3e5057a2b34, lifecycle_stage: active
        Deleted runs:
        - run_id: b13f1badbed842cf9975c023d23da300, lifecycle_stage: deleted
        All runs:
        - run_id: b13f1badbed842cf9975c023d23da300, lifecycle_stage: deleted
        - run_id: 4937823b730640d5bed9e3e5057a2b34, lifecycle_stage: active
    """
    # Using an internal function as the linter doesn't like assigning a lambda, and inlining the
    # full thing is a mess
    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().list_run_infos(
            experiment_id, run_view_type, number_to_get, order_by, next_page_token
        )

    return _paginate(pagination_wrapper_func, SEARCH_MAX_RESULTS_DEFAULT, max_results)


def _paginate(paginated_fn, max_results_per_page, max_results):
    """
    Intended to be a general use pagination utility.

    :param paginated_fn:
    :type paginated_fn: This function is expected to take in the number of results to retrieve
        per page and a pagination token, and return a PagedList object
    :param max_results_per_page:
    :type max_results_per_page: The maximum number of results to retrieve per page
    :param max_results:
    :type max_results: The maximum number of results to retrieve overall
    :return: Returns a list of entities, as determined by the paginated_fn parameter, with no more
        entities than specified by max_results
    :rtype: list[object]
    """
    all_results = []
    next_page_token = None
    while len(all_results) < max_results:
        num_to_get = max_results - len(all_results)
        if num_to_get < max_results_per_page:
            page_results = paginated_fn(num_to_get, next_page_token)
        else:
            page_results = paginated_fn(max_results_per_page, next_page_token)
        all_results.extend(page_results)
        if hasattr(page_results, "token") and page_results.token:
            next_page_token = page_results.token
        else:
            break
    return all_results


def _get_or_start_run():
    if len(_active_run_stack) > 0:
        return _active_run_stack[-1]
    return start_run()


def _get_experiment_id_from_env():
    experiment_name = env.get_env(_EXPERIMENT_NAME_ENV_VAR)
    if experiment_name is not None:
        exp = MlflowClient().get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else None
    return env.get_env(_EXPERIMENT_ID_ENV_VAR)


def _get_experiment_id():
    # TODO: Replace with None for 1.0, leaving for 0.9.1 release backcompat with existing servers
    deprecated_default_exp_id = "0"

    return (
        _active_experiment_id
        or _get_experiment_id_from_env()
        or (is_in_databricks_notebook() and get_notebook_id())
    ) or deprecated_default_exp_id


@autologging_integration("mlflow")
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for all supported integrations.

    The parameters are passed to any autologging integrations that support them.

    See the :ref:`tracking docs <automatic-logging>` for a list of supported autologging
    integrations.

    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with model artifacts during training. If ``False``,
                               input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with model artifacts during training. If ``False``, signatures are
                                 not logged. Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables all supported autologging integrations. If ``False``,
                    enables all supported autologging integrations.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.

    .. code-block:: python
        :caption: Example

        import numpy as np
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
        from sklearn.linear_model import LinearRegression

        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))

        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        # Auto log all the parameters, metrics, and artifacts
        mlflow.autolog()
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)

        # fetch the auto logged parameters and metrics for ended run
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    .. code-block:: text
        :caption: Output

        run_id: fd10a17d028c47399a55ab8741721ef7
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']
        params: {'copy_X': 'True',
                 'normalize': 'False',
                 'fit_intercept': 'True',
                 'n_jobs': 'None'}
        metrics: {'training_score': 1.0,
                  'training_rmse': 4.440892098500626e-16,
                  'training_r2_score': 1.0,
                  'training_mae': 2.220446049250313e-16,
                  'training_mse': 1.9721522630525295e-31}
        tags: {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
               'estimator_name': 'LinearRegression'}
    """
    locals_copy = locals().items()

    # Mapping of library module name to specific autolog function
    # eg: mxnet.gluon is the actual library, mlflow.gluon.autolog is our autolog function for it
    LIBRARY_TO_AUTOLOG_FN = {
        "tensorflow": tensorflow.autolog,
        "keras": keras.autolog,
        "mxnet.gluon": gluon.autolog,
        "xgboost": xgboost.autolog,
        "lightgbm": lightgbm.autolog,
        "statsmodels": statsmodels.autolog,
        "sklearn": sklearn.autolog,
        "fastai": fastai.autolog,
        "pyspark": spark.autolog,
        # TODO: Broaden this beyond pytorch_lightning as we add autologging support for more
        # Pytorch frameworks under mlflow.pytorch.autolog
        "pytorch_lightning": pytorch.autolog,
    }

    def get_autologging_params(autolog_fn):
        try:
            needed_params = list(inspect.signature(autolog_fn).parameters.keys())
            return {k: v for k, v in locals_copy if k in needed_params}
        except Exception:
            return {}

    def setup_autologging(module):
        try:
            autolog_fn = LIBRARY_TO_AUTOLOG_FN[module.__name__]
            autologging_params = get_autologging_params(autolog_fn)
            autolog_fn(**autologging_params)
            _logger.info("Autologging successfully enabled for %s.", module.__name__)
        except Exception as e:
            if _is_testing():
                # Raise unexpected exceptions in test mode in order to detect
                # errors within dependent autologging integrations
                raise
            else:
                _logger.warning(
                    "Exception raised while enabling autologging for %s: %s",
                    module.__name__,
                    str(e),
                )

    # for each autolog library (except pyspark), register a post-import hook.
    # this way, we do not send any errors to the user until we know they are using the library.
    # the post-import hook also retroactively activates for previously-imported libraries.
    for module in list(set(LIBRARY_TO_AUTOLOG_FN.keys()) - set(["pyspark"])):
        register_post_import_hook(setup_autologging, module, overwrite=True)

    # for pyspark, we activate autologging immediately, without waiting for a module import.
    # this is because on Databricks a SparkSession already exists and the user can directly
    #   interact with it, and this activity should be logged.
    try:
        autologging_params = get_autologging_params(spark.autolog)
        spark.autolog(**autologging_params)
    except ImportError as ie:
        # if pyspark isn't installed, a user could potentially install it in the middle
        #   of their session so we want to enable autologging once they do
        if "pyspark" in str(ie):
            register_post_import_hook(setup_autologging, "pyspark", overwrite=True)
    except Exception as e:
        if _is_testing():
            # Raise unexpected exceptions in test mode in order to detect
            # errors within dependent autologging integrations
            raise
        else:
            _logger.warning("Exception raised while enabling autologging for spark: %s", str(e))
