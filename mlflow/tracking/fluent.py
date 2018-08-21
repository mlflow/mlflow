"""
Internal module implementing the fluent API, allowing management of an active
MLflow run. This module is exposed to users at the top-level :py:mod:`mlflow` module.
"""

from __future__ import print_function

import atexit
import numbers
import os
import sys
import time

from mlflow.entities import Experiment, Run, SourceType
from mlflow.utils import env
from mlflow.utils.validation import _validate_run_id
from mlflow.tracking.service import get_service


_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
_active_run = None


class ActiveRun(Run):  # pylint: disable=W0223
    """Wrapper around :py:class:`mlflow.entities.Run` to allow using python `with` syntax."""
    def __init__(self, run):
        Run.__init__(self, run.info, run.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FINISHED" if exc_type is None else "FAILED"
        end_run(status)
        return exc_type is None


def start_run(run_uuid=None, experiment_id=None, source_name=None, source_version=None,
              entry_point_name=None, source_type=None, run_name=None):
    """
    Start a new MLflow run, setting it as the active run under which metrics and params
    will be logged. The return value can be used as a context manager within a ``with`` block;
    otherwise, ``end_run()`` must be called to terminate the current run. If ``run_uuid``
    is passed or the ``MLFLOW_RUN_ID`` environment variable is set, ``start_run`` attempts to
    resume a run with the specified run ID (with ``run_uuid`` taking precedence over
    ``MLFLOW_RUN_ID``), and other parameters are ignored.

    :param run_uuid: If specified, get the run with the specified UUID and log metrics
                     and params under that run. The run's end time is unset and its status
                     is set to running, but the run's other attributes remain unchanged
                     (the run's ``source_version``, ``source_type``, etc. are not changed).
    :param experiment_id: Used only when ``run_uuid`` is unspecified. ID of the experiment under
                          which to create the current run. If unspecified, the run is created under
                          a new experiment with a randomly generated name.
    :param source_name: Name of the source file or URI of the project to be associated with the run.
                        Defaults to the current file if none provided.
    :param source_version: Optional Git commit hash to associate with the run.
    :param entry_point_name: Optional name of the entry point for to the current run.
    :param source_type: Integer enum value describing the type of the run
                        ("local", "project", etc.). Defaults to
                        ``mlflow.entities.SourceType.LOCAL``.
    :return: :py:class:`mlflow.ActiveRun` object that acts as a context manager wrapping
             the run's state.
    """
    global _active_run
    if _active_run:
        raise Exception("Run with UUID %s is already active, unable to start nested "
                        "run" % _active_run.info.run_uuid)
    existing_run_uuid = run_uuid or os.environ.get(_RUN_ID_ENV_VAR, None)
    if existing_run_uuid:
        _validate_run_id(existing_run_uuid)
        active_run_obj = get_service().get_run(existing_run_uuid)
    else:
        exp_id_for_run = experiment_id or _get_experiment_id()
        active_run_obj = get_service().create_run(
            experiment_id=exp_id_for_run,
            run_name=run_name,
            source_name=source_name or _get_source_name(),
            source_version=source_version or _get_source_version(),
            entry_point_name=entry_point_name,
            source_type=source_type or _get_source_type())
    _active_run = ActiveRun(active_run_obj)
    return _active_run


def end_run(status="FINISHED"):
    global _active_run
    if _active_run:
        get_service().set_terminated(_active_run.info.run_uuid, status)
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_ID_ENV_VAR)
        _active_run = None
atexit.register(end_run)


def active_run():
    """Return the currently active ``Run``, or None if no such run exists."""
    return _active_run


def log_param(key, value):
    """
    Log the passed-in parameter under the current run, creating a run if necessary.

    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_uuid
    get_service().log_param(run_id, key, value)


def log_metric(key, value):
    """
    Log the passed-in metric under the current run, creating a run if necessary.

    :param key: Metric name (string).
    :param value: Metric value (float).
    """
    if not isinstance(value, numbers.Number):
        print("WARNING: The metric {}={} was not logged because the value is not a number.".format(
            key, value), file=sys.stderr)
        return
    run_id = _get_or_start_run().info.run_uuid
    get_service().log_metric(run_id, key, value, int(time.time()))


def log_artifact(local_path, artifact_path=None):
    """Log a local file or directory as an artifact of the currently active run."""
    artifact_uri = _get_or_start_run().info.artifact_uri
    get_service().log_artifact(artifact_uri, local_path, artifact_path)


def log_artifacts(local_dir, artifact_path=None):
    """Log all the contents of a local directory as artifacts of the run."""
    artifact_uri = _get_or_start_run().info.artifact_uri
    get_service().log_artifacts(artifact_uri, local_dir, artifact_path)


def create_experiment(name, artifact_location=None):
    return get_service().create_experiment(name, artifact_location)


def get_artifact_uri():
    """
    Return the artifact URI of the currently active run. Calls to ``log_artifact`` and
    ``log_artifacts`` write artifact(s) to subdirectories of the returned URI.
    """
    return _get_or_start_run().info.artifact_uri


def _get_or_start_run():
    if _active_run:
        return _active_run
    return start_run()


def _get_main_file():
    if len(sys.argv) > 0:
        return sys.argv[0]
    return None


def _get_source_name():
    main_file = _get_main_file()
    if main_file is not None:
        return main_file
    return "<console>"


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return _get_git_commit(main_file)
    return None


def _get_source_type():
    return SourceType.LOCAL


def _get_experiment_id():
    return int(env.get_env(_EXPERIMENT_ID_ENV_VAR) or Experiment.DEFAULT_EXPERIMENT_ID)


def _get_git_commit(path):
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound, NoSuchPathError
    except ImportError as e:
        print("Notice: failed to import Git (the git executable is probably not on your PATH),"
              " so Git SHA is not available. Error: %s" % e, file=sys.stderr)
        return None
    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return None
