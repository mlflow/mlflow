from __future__ import print_function

import atexit
import numbers
import os
import sys
import time

from six.moves import urllib

from mlflow.entities.experiment import Experiment
from mlflow.entities.param import Param
from mlflow.entities.metric import Metric
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType
from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils import env


_RUN_NAME_ENV_VAR = "MLFLOW_RUN_NAME"
_DEFAULT_USER_ID = "unknown"
_LOCAL_FS_URI_PREFIX = "file:///"
_REMOTE_URI_PREFIX = "http://"
_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_active_run = None
_tracking_uri = None


def _get_user_id():
    """ Gets the ID of the user for the current run. """
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID


def set_tracking_uri(uri):
    """
    Sets the tracking server URI to the passed-in value. Note that this does not affect the
    currently active run (if one exists), but will take effect for any successive runs.
    """
    global _tracking_uri
    _tracking_uri = uri


def get_tracking_uri():
    """
    Returns the current tracking URI. Note that this may not correspond to the tracking URI of
    the currently active run, since the tracking URI may be updated via `set_tracking_uri`.
    :return: the tracking URI
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return os.path.abspath("./mlruns")


def _is_local_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == '' or scheme == 'file'


def _is_http_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == '' or scheme == 'http'


def _get_file_store(store_uri):
    path = urllib.parse.urlparse(store_uri).path
    return FileStore(path)


def _get_rest_store(store_uri):
    return RestStore(store_uri)


def _get_store():
    store_uri = get_tracking_uri()
    # Default: if URI hasn't been set, return a FileStore
    if store_uri is None:
        return FileStore()
    # Pattern-match on the URI
    if _is_local_uri(store_uri):
        return _get_file_store(store_uri)
    if _is_http_uri(store_uri):
        return _get_rest_store(store_uri)

    raise Exception("Tracking URI must be a local filesystem URI of the form '%s...' or a "
                    "remote URI of the form '%s...'. Please update the tracking URI via "
                    "mlflow.set_tracking_uri" % (_LOCAL_FS_URI_PREFIX, _REMOTE_URI_PREFIX))


class ActiveRun(object):
    """
    Class representing an active run. Has a reference to the store to which state for the run
    (e.g. run metadata, metrics, parameters, and artifacts) should be persisted.

    Contains methods for logging metrics, parameters, etc under the current run.

    :param run_info: RunInfo describing the active run. A corresponding `Run` object is assumed to
                     already be persisted with state "running" in `store`.
    :param store: Backend store to which the current run should persist state updates.
    """
    def __init__(self, run_info, store):
        self.store = store
        self.run_info = run_info
        if run_info.artifact_uri:
            self.artifact_repo = ArtifactRepository.from_artifact_uri(run_info.artifact_uri)
        else:
            self.artifact_repo = _get_legacy_artifact_repo(store, run_info)
        global _active_run
        _active_run = self

    def set_terminated(self, status):
        self.run_info = self.store.update_run_info(
            self.run_info.run_uuid, run_status=RunStatus.from_string(status),
            end_time=_get_unix_timestamp())

    def log_metric(self, metric):
        self.store.log_metric(self.run_info.run_uuid, metric)

    def log_param(self, param):
        self.store.log_param(self.run_info.run_uuid, param)

    def log_artifact(self, local_path, artifact_path=None):
        self.artifact_repo.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        self.artifact_repo.log_artifacts(local_dir, artifact_path)

    def get_artifact_uri(self):
        return self.artifact_repo.artifact_uri

    def get_run(self):
        return self.store.get_run(self.run_info.run_uuid)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FINISHED" if exc_type is None else "FAILED"
        self.set_terminated(status)
        global _active_run
        _active_run = None
        return exc_type is None


def list_experiments():
    """
    Returns a list of all experiments
    """
    return _get_store().list_experiments()


def create_experiment(experiment_name):
    """
    Creates an experiment with the specified name and returns its ID.
    """
    if experiment_name is None or experiment_name == "":
        raise Exception("Invalid experiment name '%s'" % experiment_name)
    return _get_store().create_experiment(experiment_name)


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


def _get_unix_timestamp():
    return int(time.time() * 1000)


def _do_start_run(run_uuid=None, experiment_id=None, source_name=None, source_version=None,
                  entry_point_name=None, source_type=None):
    store = _get_store()

    if run_uuid is not None:
        existing_run = store.get_run(run_uuid)
        if existing_run is None:
            raise Exception("Could not start run with UUID %s - no such run found." % run_uuid)
        updated_info = store.update_run_info(
            run_uuid=run_uuid, run_status=RunStatus.RUNNING, end_time=None)
        return ActiveRun(updated_info, store)

    # Get experiment ID for run
    exp_id_for_run = experiment_id or _get_experiment_id()
    run = store.create_run(experiment_id=exp_id_for_run, user_id=_get_user_id(), run_name=None,
                           source_type=(source_type or _get_source_type()),
                           source_name=(source_name or _get_source_name()),
                           entry_point_name=entry_point_name,
                           start_time=_get_unix_timestamp(),
                           source_version=(source_version or _get_source_version()), tags=[])
    return ActiveRun(run.info, store)


def start_run(run_uuid=None, experiment_id=None, source_name=None, source_version=None,
              entry_point_name=None, source_type=None):
    """
    Start a new MLflow run, setting it as the active run under which metrics and params
    will be logged. The return value can be used as a context manager within a `with` block;
    otherwise, `end_run()` must be called to terminate the current run.

    Note that if a run is currently in progress (i.e., start_run has been called without an
    end_run, or you are running from "mlflow run"), then this function will return the
    existing run.

    :param run_uuid: If specified, gets the run with the specified UUID and logs metrics
                     and params under that run. The run's end time will be unset and its status
                     will be set to running, but the run's other attributes will remain unchanged
                     (the run's source_version, source_type, etc will not be changed).
    :param experiment_id: Only used when run_uuid is unspecified. ID of the experiment under which
                          to create the current run. If unspecified, the run will be created under
                          a new experiment with a randomly-generated name
    :param source_name: Name of the source file or URI of the project to be associated with the run.
                        Defaults to the current file if none provided.
    :param source_version: Optional Git commit hash to associate with the run.
    :param entry_point_name: Optional name of the entry point for to the current run.
    :param source_type: Integer enum value describing the type of the run ("local", "project", etc).
                        Defaults to mlflow.entities.source_type.SourceType.LOCAL.
    :return: A :class:`ActiveRun` object that acts as a context manager wrapping the run's state
    """
    global _active_run
    if _active_run:
        return _active_run
    if _RUN_NAME_ENV_VAR not in os.environ:
        return _do_start_run(run_uuid, experiment_id, source_name, source_version,
                             entry_point_name, source_type)

    # Load an existing run ID from the environment
    existing_run_uuid = os.environ[_RUN_NAME_ENV_VAR]
    store = _get_store()
    run = store.get_run(existing_run_uuid)
    # If we were able to find an existing run with the specified ID, create an ActiveRun with
    # that ID and update the global _active_run
    # TODO: This doesn't play well with the atexit.register(end_run) call; specifically each
    # time a process with the current run ID exits, the run will be marked as terminated.
    _active_run = ActiveRun(run.info, store)
    return _active_run


def _get_or_start_run():
    return start_run()


def end_run(status="FINISHED"):
    global _active_run
    if _active_run:
        _active_run.set_terminated(status)
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_NAME_ENV_VAR)
        _active_run = None


def active_run():
    """ Returns the currently active `Run`, or None if no such run exists. """
    if _active_run:
        return _active_run.get_run()
    else:
        return None


def log_param(key, value):
    """
    Logs the passed-in parameter under the current run, creating a run if necessary.
    :param key: Parameter name (string)
    :param value: Parameter value (string)
    """  
    _get_or_start_run().log_param(Param(key, str(value)))


def log_metric(key, value):
    """
    Logs the passed-in metric under the current run, creating a run if necessary.
    :param key: Metric name (string)
    :param value: Metric value (float)
    """
    if not isinstance(value, numbers.Number):
        print("WARNING: The metric {}={} was not logged because the value is not a number.".format(
            key, value), file=sys.stderr)
        return
    _get_or_start_run().log_metric(Metric(key, value, int(time.time())))


def log_artifact(local_path, artifact_path=None):
    """Log a local file or directory as an artifact of the currently active run."""
    _get_or_start_run().log_artifact(local_path, artifact_path)


def log_artifacts(local_dir, artifact_path=None):
    """Log all the contents of a local directory as artifacts of the run."""
    _get_or_start_run().log_artifacts(local_dir, artifact_path)


def get_artifact_uri():
    """
    Returns the artifact URI of the currently active run. Calls to `log_artifact`, `log_artifacts`
    will write artifact(s) to subdirectories of the returned URI.
    """
    return _get_or_start_run().get_artifact_uri()


atexit.register(end_run)


def _get_model_log_dir(model_name, run_id):
    if not run_id:
        raise Exception("Must specify a run_id to get logging directory for a model.")
    run = _get_store().get_run(run_id)
    if run.info.artifact_uri:
        artifact_repo = ArtifactRepository.from_artifact_uri(run.info.artifact_uri)
    else:
        artifact_repo = _get_legacy_artifact_repo(_get_store(), run.info)
    return artifact_repo.download_artifacts(model_name)


def _get_legacy_artifact_repo(file_store, run_info):
    # TODO(aaron) Remove this once everyone locally only has runs from after
    # the introduction of "artifact_uri".
    uri = os.path.join(file_store.root_directory, str(run_info.experiment_id),
                       run_info.run_uuid, "artifacts")
    return ArtifactRepository.from_artifact_uri(uri)


def _get_git_commit(path):
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound
    except ImportError as e:
        print("Notice: failed to import git (the git executable is probably not on your PATH),"
              " so git sha will not be available. Error: %s" % e, file=sys.stderr)
        return None 
    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError):
        return None
