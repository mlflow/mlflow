"""
Internal module implementing the fluent API, allowing management of an active
MLflow run. This module is exposed to users at the top-level :py:mod:`mlflow` module.
"""

from __future__ import print_function

import os

import atexit
import time
import logging

import mlflow.tracking.utils
from mlflow.entities import Experiment, Run, SourceType, RunStatus, Param, RunTag, Metric
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking import context
from mlflow.utils import env
from mlflow.utils.databricks_utils import is_in_databricks_notebook, get_notebook_id
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_SOURCE_TYPE, MLFLOW_SOURCE_NAME, \
    MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_PARENT_RUN_ID
from mlflow.utils.validation import _validate_run_id

_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_EXPERIMENT_NAME_ENV_VAR = "MLFLOW_EXPERIMENT_NAME"
_RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
_active_run_stack = []
_active_experiment_id = None


_logger = logging.getLogger(__name__)


def set_experiment(experiment_name):
    """
    Set given experiment as active experiment. If experiment does not exist, create an experiment
    with provided name.

    :param experiment_name: Name of experiment to be activated.
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
            " experiment to create a new one." % experiment.name)
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


def start_run(run_uuid=None, experiment_id=None, source_name=None, source_version=None,
              entry_point_name=None, source_type=None, run_name=None, nested=False):
    """
    Start a new MLflow run, setting it as the active run under which metrics and parameters
    will be logged. The return value can be used as a context manager within a ``with`` block;
    otherwise, you must call ``end_run()`` to terminate the current run.

    If you pass a ``run_uuid`` or the ``MLFLOW_RUN_ID`` environment variable is set,
    ``start_run`` attempts to resume a run with the specified run ID and
    other parameters are ignored. ``run_uuid`` takes precedence over ``MLFLOW_RUN_ID``.

    :param run_uuid: If specified, get the run with the specified UUID and log parameters
                     and metrics under that run. The run's end time is unset and its status
                     is set to running, but the run's other attributes (``source_version``,
                     ``source_type``, etc.) are not changed.
    :param experiment_id: ID of the experiment under which to create the current run (applicable
                          only when ``run_uuid`` is not specified). If ``experiment_id`` argument
                          is unspecified, will look for valid experiment in the following order:
                          activated using ``set_experiment``, ``MLFLOW_EXPERIMENT_ID`` env variable,
                          or the default experiment.
    :param source_name: Name of the source file or URI of the project to be associated with the run.
                        If none provided defaults to the current file.
    :param source_version: Optional Git commit hash to associate with the run.
    :param entry_point_name: Optional name of the entry point for the current run.
    :param source_type: Integer :py:class:`mlflow.entities.SourceType` describing the type
                        of the run ("local", "project", etc.). Defaults to
                        :py:class:`mlflow.entities.SourceType.LOCAL` ("local").
    :param run_name: Name of new run. Used only when ``run_uuid`` is unspecified.
    :param nested: Parameter which must be set to ``True`` to create nested runs.
    :return: :py:class:`mlflow.ActiveRun` object that acts as a context manager wrapping
             the run's state.
    """
    global _active_run_stack
    if len(_active_run_stack) > 0 and not nested:
        raise Exception(("Run with UUID {} is already active. To start a nested " +
                        "run call start_run with nested=True").format(
            _active_run_stack[0].info.run_uuid))
    existing_run_uuid = run_uuid or os.environ.get(_RUN_ID_ENV_VAR, None)
    if existing_run_uuid:
        _validate_run_id(existing_run_uuid)
        active_run_obj = MlflowClient().get_run(existing_run_uuid)
        if active_run_obj.info.lifecycle_stage == LifecycleStage.DELETED:
            raise MlflowException("Cannot start run with ID {} because it is in the "
                                  "deleted state.".format(existing_run_uuid))
    else:
        if len(_active_run_stack) > 0:
            parent_run_id = _active_run_stack[-1].info.run_uuid
        else:
            parent_run_id = None

        exp_id_for_run = experiment_id if experiment_id is not None else _get_experiment_id()

        user_specified_tags = {}
        if parent_run_id is not None:
            user_specified_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        if source_name is not None:
            user_specified_tags[MLFLOW_SOURCE_NAME] = source_name
        if source_type is not None:
            user_specified_tags[MLFLOW_SOURCE_TYPE] = SourceType.to_string(source_type)
        if source_version is not None:
            user_specified_tags[MLFLOW_GIT_COMMIT] = source_version
        if entry_point_name is not None:
            user_specified_tags[MLFLOW_PROJECT_ENTRY_POINT] = entry_point_name

        tags = context.resolve_tags(user_specified_tags)

        active_run_obj = MlflowClient().create_run(
            experiment_id=exp_id_for_run,
            run_name=run_name,
            tags=tags
        )

    _active_run_stack.append(ActiveRun(active_run_obj))
    return _active_run_stack[-1]


def end_run(status=RunStatus.to_string(RunStatus.FINISHED)):
    """End an active MLflow run (if there is one)."""
    global _active_run_stack
    if len(_active_run_stack) > 0:
        MlflowClient().set_terminated(_active_run_stack[-1].info.run_uuid, status)
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_ID_ENV_VAR)
        _active_run_stack.pop()


atexit.register(end_run)


def active_run():
    """Get the currently active ``Run``, or None if no such run exists."""
    return _active_run_stack[-1] if len(_active_run_stack) > 0 else None


def log_param(key, value):
    """
    Log a parameter under the current run, creating a run if necessary.

    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_uuid
    MlflowClient().log_param(run_id, key, value)


def set_tag(key, value):
    """
    Set a tag under the current run, creating a run if necessary.

    :param key: Tag name (string)
    :param value: Tag value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_uuid
    MlflowClient().set_tag(run_id, key, value)


def log_metric(key, value):
    """
    Log a metric under the current run, creating a run if necessary.

    :param key: Metric name (string).
    :param value: Metric value (float).
    """
    run_id = _get_or_start_run().info.run_uuid
    MlflowClient().log_metric(run_id, key, value, int(time.time()))


def log_metrics(metrics):
    """
    Log multiple metrics for the current run, starting a run if no runs are active.
    :param metrics: Dictionary of metric_name: String -> value: Float
    :returns: None
    """
    run_id = _get_or_start_run().info.run_uuid
    timestamp = int(time.time())
    metrics_arr = [Metric(key, value, timestamp) for key, value in metrics.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=[])


def log_params(params):
    """
    Log a batch of params for the current run, starting a run if no runs are active.
    :param params: Dictionary of param_name: String -> value: (String, but will be string-ified if
                   not)
    :returns: None
    """
    run_id = _get_or_start_run().info.run_uuid
    params_arr = [Param(key, value) for key, value in params.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=params_arr, tags=[])


def set_tags(tags):
    """
    Log a batch of tags for the current run, starting a run if no runs are active.
    :param tags: Dictionary of tag_name: String -> value: (String, but will be string-ified if
                 not)
    :returns: None
    """
    run_id = _get_or_start_run().info.run_uuid
    tags_arr = [RunTag(key, value) for key, value in tags.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=[], tags=tags_arr)


def log_artifact(local_path, artifact_path=None):
    """
    Log a local file or directory as an artifact of the currently active run.

    :param local_path: Path to the file to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
    """
    run_id = _get_or_start_run().info.run_uuid
    MlflowClient().log_artifact(run_id, local_path, artifact_path)


def log_artifacts(local_dir, artifact_path=None):
    """
    Log all the contents of a local directory as artifacts of the run.

    :param local_dir: Path to the directory of files to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
    """
    run_id = _get_or_start_run().info.run_uuid
    MlflowClient().log_artifacts(run_id, local_dir, artifact_path)


def create_experiment(name, artifact_location=None):
    """
    Create an experiment.

    :param name: The experiment name. Must be unique.
    :param artifact_location: The location to store run artifacts.
                              If not provided, the server picks an appropriate default.
    :return: Integer ID of the created experiment.
    """
    return MlflowClient().create_experiment(name, artifact_location)


def get_artifact_uri(artifact_path=None):
    """
    Get the absolute URI of the specified artifact in the currently active run.
    If `path` is not specified, the artifact root URI of the currently active
    run will be returned; calls to ``log_artifact`` and ``log_artifacts`` write
    artifact(s) to subdirectories of the artifact root URI.

    :param artifact_path: The run-relative artifact path for which to obtain an absolute URI.
                          For example, "path/to/artifact". If unspecified, the artifact root URI
                          for the currently active run will be returned.
    :return: An *absolute* URI referring to the specified artifact or the currently adtive run's
             artifact root. For example, if an artifact path is provided and the currently active
             run uses an S3-backed store, this may be a uri of the form
             ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
             is not provided and the currently active run uses an S3-backed store, this may be a
             URI of the form ``s3://<bucket_name>/path/to/artifact/root``.
    """
    return mlflow.tracking.utils.get_artifact_uri(
        run_id=_get_or_start_run().info.run_uuid, artifact_path=artifact_path)


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
    return int(_active_experiment_id or
               _get_experiment_id_from_env() or
               (is_in_databricks_notebook() and get_notebook_id()) or
               Experiment.DEFAULT_EXPERIMENT_ID)
