"""
Internal module implementing the fluent API, allowing management of an active
MLflow run. This module is exposed to users at the top-level :py:mod:`mlflow` module.
"""

from __future__ import print_function

import os

import atexit
import time
import logging
import numpy as np
import pandas as pd

from mlflow.entities import Run, RunStatus, Param, RunTag, Metric, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking import artifact_utils
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import env
from mlflow.utils.databricks_utils import is_in_databricks_notebook, get_notebook_id
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.validation import _validate_run_id

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


def start_run(run_id=None, experiment_id=None, run_name=None, nested=False):
    """
    Start a new MLflow run, setting it as the active run under which metrics and parameters
    will be logged. The return value can be used as a context manager within a ``with`` block;
    otherwise, you must call ``end_run()`` to terminate the current run.

    If you pass a ``run_id`` or the ``MLFLOW_RUN_ID`` environment variable is set,
    ``start_run`` attempts to resume a run with the specified run ID and
    other parameters are ignored. ``run_id`` takes precedence over ``MLFLOW_RUN_ID``.

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
    :param nested: Controls whether run is nested in parent run. ``True`` creates a nest run.
    :return: :py:class:`mlflow.ActiveRun` object that acts as a context manager wrapping
             the run's state.
    """
    global _active_run_stack
    # back compat for int experiment_id
    experiment_id = str(experiment_id) if isinstance(experiment_id, int) else experiment_id
    if len(_active_run_stack) > 0 and not nested:
        raise Exception(("Run with UUID {} is already active. To start a nested " +
                        "run, call start_run with nested=True").format(
            _active_run_stack[0].info.run_id))
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
        if (_active_experiment_id is not None and
                _active_experiment_id != active_run_obj.info.experiment_id):
            raise MlflowException("Cannot start run with ID {} because active run ID "
                                  "does not match environment run ID. Make sure --experiment-name "
                                  "or --experiment-id matches experiment set with "
                                  "set_experiment(), or just use command-line "
                                  "arguments".format(existing_run_id))
        # Check to see if current run isn't deleted
        if active_run_obj.info.lifecycle_stage == LifecycleStage.DELETED:
            raise MlflowException("Cannot start run with ID {} because it is in the "
                                  "deleted state.".format(existing_run_id))
    else:
        if len(_active_run_stack) > 0:
            parent_run_id = _active_run_stack[-1].info.run_id
        else:
            parent_run_id = None

        exp_id_for_run = experiment_id if experiment_id is not None else _get_experiment_id()

        user_specified_tags = {}
        if parent_run_id is not None:
            user_specified_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        if run_name is not None:
            user_specified_tags[MLFLOW_RUN_NAME] = run_name

        tags = context_registry.resolve_tags(user_specified_tags)

        active_run_obj = MlflowClient().create_run(
            experiment_id=exp_id_for_run,
            tags=tags
        )

    _active_run_stack.append(ActiveRun(active_run_obj))
    return _active_run_stack[-1]


def end_run(status=RunStatus.to_string(RunStatus.FINISHED)):
    """End an active MLflow run (if there is one)."""
    global _active_run_stack
    if len(_active_run_stack) > 0:
        MlflowClient().set_terminated(_active_run_stack[-1].info.run_id, status)
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_ID_ENV_VAR)
        _active_run_stack.pop()


atexit.register(end_run)


def active_run():
    """Get the currently active ``Run``, or None if no such run exists."""
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
    """
    return MlflowClient().get_run(run_id)


def log_param(key, value):
    """
    Log a parameter under the current run, creating a run if necessary.

    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_param(run_id, key, value)


def set_tag(key, value):
    """
    Set a tag under the current run, creating a run if necessary.

    :param key: Tag name (string)
    :param value: Tag value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().set_tag(run_id, key, value)


def delete_tag(key):
    """
    Delete a tag from a run. This is irreversible.

    :param key: Name of the tag
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().delete_tag(run_id, key)


def log_metric(key, value, step=None):
    """
    Log a metric under the current run, creating a run if necessary.

    :param key: Metric name (string).
    :param value: Metric value (float). Note that some special values such as +/- Infinity may be
                  replaced by other values depending on the store. For example, sFor example, the
                  SQLAlchemy store replaces +/- Inf with max / min float values.
    :param step: Metric step (int). Defaults to zero if unspecified.
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_metric(run_id, key, value, int(time.time() * 1000), step or 0)


def log_metrics(metrics, step=None):
    """
    Log multiple metrics for the current run, starting a run if no runs are active.

    :param metrics: Dictionary of metric_name: String -> value: Float. Note that some special values
                    such as +/- Infinity may be replaced by other values depending on the store.
                    For example, sql based store may replace +/- Inf with max / min float values.
    :param step: A single integer step at which to log the specified
                 Metrics. If unspecified, each metric is logged at step zero.

    :returns: None
    """
    run_id = _get_or_start_run().info.run_id
    timestamp = int(time.time() * 1000)
    metrics_arr = [Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=[])


def log_params(params):
    """
    Log a batch of params for the current run, starting a run if no runs are active.

    :param params: Dictionary of param_name: String -> value: (String, but will be string-ified if
                   not)
    :returns: None
    """
    run_id = _get_or_start_run().info.run_id
    params_arr = [Param(key, str(value)) for key, value in params.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=params_arr, tags=[])


def set_tags(tags):
    """
    Log a batch of tags for the current run, starting a run if no runs are active.

    :param tags: Dictionary of tag_name: String -> value: (String, but will be string-ified if
                 not)
    :returns: None
    """
    run_id = _get_or_start_run().info.run_id
    tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
    MlflowClient().log_batch(run_id=run_id, metrics=[], params=[], tags=tags_arr)


def log_artifact(local_path, artifact_path=None):
    """
    Log a local file or directory as an artifact of the currently active run.

    :param local_path: Path to the file to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_artifact(run_id, local_path, artifact_path)


def log_artifacts(local_dir, artifact_path=None):
    """
    Log all the contents of a local directory as artifacts of the run.

    :param local_dir: Path to the directory of files to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_artifacts(run_id, local_dir, artifact_path)


def get_experiment(experiment_id):
    """
    Retrieve an experiment by experiment_id from the backend store

    :param experiment_id: The experiment ID returned from ``create_experiment``.
    :return: :py:class:`mlflow.entities.Experiment`
    """
    return MlflowClient().get_experiment(experiment_id)


def get_experiment_by_name(name):
    """
    Retrieve an experiment by experiment name from the backend store

    :param name: The experiment name.
    :return: :py:class:`mlflow.entities.Experiment`
    """
    return MlflowClient().get_experiment_by_name(name)


def create_experiment(name, artifact_location=None):
    """
    Create an experiment.

    :param name: The experiment name. Must be unique.
    :param artifact_location: The location to store run artifacts.
                              If not provided, the server picks an appropriate default.
    :return: Integer ID of the created experiment.
    """
    return MlflowClient().create_experiment(name, artifact_location)


def delete_experiment(experiment_id):
    """
    Delete an experiment from the backend store.

    :param experiment_id: The experiment ID returned from ``create_experiment``.
    """
    MlflowClient().delete_experiment(experiment_id)


def delete_run(run_id):
    """
    Deletes a run with the given ID.

    :param run_id: Unique identifier for the run to delete.
    """
    MlflowClient().delete_run(run_id)


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
    return artifact_utils.get_artifact_uri(run_id=_get_or_start_run().info.run_id,
                                           artifact_path=artifact_path)


def search_runs(experiment_ids=None, filter_string="", run_view_type=ViewType.ACTIVE_ONLY,
                max_results=SEARCH_MAX_RESULTS_PANDAS, order_by=None):
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
    """
    if not experiment_ids:
        experiment_ids = _get_experiment_id()
    runs = _get_paginated_runs(experiment_ids, filter_string, run_view_type, max_results,
                               order_by)
    info = {'run_id': [], 'experiment_id': [],
            'status': [], 'artifact_uri': [],
            'start_time': [], 'end_time': []}
    params, metrics, tags = ({}, {}, {})
    PARAM_NULL, METRIC_NULL, TAG_NULL = (None, np.nan, None)
    for i, run in enumerate(runs):
        info['run_id'].append(run.info.run_id)
        info['experiment_id'].append(run.info.experiment_id)
        info['status'].append(run.info.status)
        info['artifact_uri'].append(run.info.artifact_uri)
        info['start_time'].append(pd.to_datetime(run.info.start_time, unit="ms", utc=True))
        info['end_time'].append(pd.to_datetime(run.info.end_time, unit="ms", utc=True))

        # Params
        param_keys = set(params.keys())
        for key in param_keys:
            if key in run.data.params:
                params[key].append(run.data.params[key])
            else:
                params[key].append(PARAM_NULL)
        new_params = set(run.data.params.keys()) - param_keys
        for p in new_params:
            params[p] = [PARAM_NULL]*i  # Fill in null values for all previous runs
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
            metrics[m] = [METRIC_NULL]*i
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
            tags[t] = [TAG_NULL]*i
            tags[t].append(run.data.tags[t])

    data = {}
    data.update(info)
    for key in metrics:
        data['metrics.' + key] = metrics[key]
    for key in params:
        data['params.' + key] = params[key]
    for key in tags:
        data['tags.' + key] = tags[key]
    return pd.DataFrame(data)


def _get_paginated_runs(experiment_ids, filter_string, run_view_type, max_results,
                        order_by):
    all_runs = []
    next_page_token = None
    while(len(all_runs) < max_results):
        runs_to_get = max_results-len(all_runs)
        if runs_to_get < NUM_RUNS_PER_PAGE_PANDAS:
            runs = MlflowClient().search_runs(experiment_ids, filter_string, run_view_type,
                                              runs_to_get, order_by, next_page_token)
        else:
            runs = MlflowClient().search_runs(experiment_ids, filter_string, run_view_type,
                                              NUM_RUNS_PER_PAGE_PANDAS, order_by, next_page_token)
        all_runs.extend(runs)
        if hasattr(runs, 'token') and runs.token != '' and runs.token is not None:
            next_page_token = runs.token
        else:
            break
    return all_runs


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

    return (_active_experiment_id or
            _get_experiment_id_from_env() or
            (is_in_databricks_notebook() and get_notebook_id())) or deprecated_default_exp_id
