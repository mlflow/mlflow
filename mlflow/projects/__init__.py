"""APIs for running MLflow Projects locally or remotely."""

from __future__ import print_function

import json
import os
import subprocess

from mlflow.projects._project_spec import Project
from mlflow.projects.utils import _load_project, _get_conda_env_name, _fetch_project, _get_work_dir, _expand_uri, ExecutionException

from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
import mlflow.tracking as tracking


from mlflow.utils import process
from mlflow.utils.logging_utils import eprint

# TODO: this should be restricted to just Git repos and not S3 and stuff like that
# Environment variable indicating a path to a conda installation. MLflow will default to running
# "conda" if unset
MLFLOW_CONDA = "MLFLOW_CONDA"
MLFLOW_ENTRY_POINT_MODE = "_local_run_entry_point"

def _run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
         mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
         use_temp_cwd=False, storage_dir=None, block=True, run_id=None):
    exp_id = experiment_id or tracking._get_experiment_id()
    if mode is None or mode == "local":
        return _run_local(
            uri=uri, entry_point=entry_point, version=version, parameters=parameters,
            experiment_id=exp_id, use_conda=use_conda, use_temp_cwd=use_temp_cwd,
            storage_dir=storage_dir, git_username=git_username, git_password=git_password,
            block=block, run_id=run_id)
    if mode == "databricks":
        from mlflow.projects.databricks import run_databricks
        return run_databricks(
            uri=uri, entry_point=entry_point, version=version, parameters=parameters,
            experiment_id=exp_id, cluster_spec=cluster_spec, git_username=git_username,
            git_password=git_password)
    supported_modes = ["local", "databricks"]
    raise ExecutionException("Got unsupported execution mode %s. Supported "
                             "values: %s" % (mode, supported_modes))


def run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
        mode=None, cluster_spec=None, git_username=None, git_password=None,
        use_conda=True, use_temp_cwd=False, storage_dir=None, block=True, run_id=None):
    """
    Run an MLflow project from the given URI.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.

    Raises:
      `mlflow.projects.ExecutionException` if a run launched in blocking mode is unsuccessful.

    :param uri: URI of project to run. Expected to be either a relative/absolute local filesystem
                path or a git repository URI (e.g. https://github.com/databricks/mlflow-example)
                pointing to a project directory containing an MLproject file.
    :param entry_point: Entry point to run within the project. If no entry point with the specified
                        name is found, attempts to run the project file `entry_point` as a script,
                        using "python" to run .py files and the default shell (specified by
                        environment variable $SHELL) to run .sh files.
    :param experiment_id: ID of experiment under which to launch the run.
    :param mode: Execution mode for the run. Can be set to "local" or "databricks".
    :param cluster_spec: Path to JSON file describing the cluster to use when launching a run on
                         Databricks.
    :param git_username: Username for HTTP(S) authentication with Git.
    :param git_password: Password for HTTP(S) authentication with Git.
    :param use_conda: If True (the default), creates a new Conda environment for the run and
                      installs project dependencies within that environment. Otherwise, runs the
                      project in the current environment without installing any project
                      dependencies.
    :param use_temp_cwd: Only used if `mode` is "local" and `uri` is a local directory.
                         If True, copies project to a temporary working directory before running it.
                         Otherwise (the default), runs project using `uri` (the project's path) as
                         the working directory.
    :param storage_dir: Only used if `mode` is local. MLflow will download artifacts from
                        distributed URIs passed to parameters of type 'path' to subdirectories of
                        storage_dir.
    :param block: Whether or not to block while waiting for a run to complete. Defaults to True.
                  Note that if `block` is False and mode is "local", this method will return, but
                  the current process will block when exiting until the local run completes.
                  If the current process is interrupted, any asynchronous runs launched via this
                  method will be terminated.
    :param run_id: Note: this argument is used internally by the MLflow project APIs and should
                   not be specified. If specified, the given run ID will be used instead of
                   creating a new run.
    :return: A `SubmittedRun` exposing information (e.g. run ID) about the launched run. Note that
             the returned `SubmittedRun` is not thread-safe.
    """
    if mode == MLFLOW_ENTRY_POINT_MODE:
        from mlflow.projects.local import _run_and_monitor_entry_point
        _run_and_monitor_entry_point(
            uri=uri, entry_point=entry_point, parameters=parameters, storage_dir=storage_dir,
            use_conda=use_conda)
        return None

    submitted_run_obj = _run(uri=uri, entry_point=entry_point, version=version,
                             parameters=parameters,
                             experiment_id=experiment_id,
                             mode=mode, cluster_spec=cluster_spec, git_username=git_username,
                             git_password=git_password, use_conda=use_conda,
                             use_temp_cwd=use_temp_cwd, storage_dir=storage_dir, block=block,
                             run_id=run_id)
    if block:
        if not submitted_run_obj.wait():
            raise ExecutionException("=== Run (%s, MLflow run id: %s) was unsuccessful ===" %
                                     (submitted_run_obj.describe(), submitted_run_obj.run_id))
    return submitted_run_obj


def _run_local(uri, entry_point, version, parameters, experiment_id, use_conda, use_temp_cwd,
               storage_dir, git_username, git_password, block, run_id):
    """
    Run an MLflow project from the given URI in a new directory.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.
    """
    eprint("=== Fetching project from %s ===" % uri)

    # Get the working directory to use for running the project & download it there
    work_dir = _get_work_dir(uri, use_temp_cwd)
    eprint("=== Work directory for this run: %s ===" % work_dir)
    expanded_uri = _expand_uri(uri)
    _fetch_project(expanded_uri, version, work_dir, git_username, git_password)

    # Load the MLproject file
    if not os.path.isfile(os.path.join(work_dir, "MLproject")):
        raise ExecutionException("No MLproject file found in %s" % uri)
    project = _load_project(work_dir, uri)
    return _run_project(
        project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id, run_id,
        block)


def _conda_executable():
    """
    Returns path to a conda executable. Configurable via the mlflow.projects.MLFLOW_CONDA
    environment variable.
    """
    return os.environ.get(MLFLOW_CONDA, "conda")


def _maybe_create_conda_env(conda_env_path):
    conda_env = _get_conda_env_name(conda_env_path)
    conda_path = _conda_executable()
    try:
        process.exec_cmd([conda_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException("Could not find conda executable at {0}. "
                                 "Please ensure conda is installed as per the instructions "
                                 "at https://conda.io/docs/user-guide/install/index.html. You may "
                                 "also configure MLflow to look for a specific conda executable "
                                 "by setting the {1} environment variable to the path of the conda "
                                 "executable".format(conda_path, MLFLOW_CONDA))
    (_, stdout, _) = process.exec_cmd([conda_path, "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]

    conda_action = 'create'
    if conda_env not in env_names:
        eprint('=== Creating conda environment %s ===' % conda_env)
        process.exec_cmd([conda_path, "env", conda_action, "-n", conda_env, "--file",
                          conda_env_path], stream_output=True)


def _launch_local_run(
        uri, run_id, entry_point, parameters, work_dir, env_map, use_conda, storage_dir,
        stream_output):
    """
    Runs an entry point by launching its command in a subprocess, updating the tracking server with
    the run's exit status.

    :param command: Entry point command to execute
    :param work_dir: Working directory to use when executing `command`
    :param env_map: Dict of environment variable key-value pairs to set in the process for `command`
    :return `SubmittedRun` corresponding to the launched run.
    """
    from mlflow.projects.submitted_run import LocalSubmittedRun
    command = _load_project(work_dir, uri)
    mlflow_run_arr = ["mlflow", "run", work_dir, "-e", entry_point, "-m", MLFLOW_ENTRY_POINT_MODE]
    if storage_dir is not None:
        mlflow_run_arr.extend(["--storage-dir", storage_dir])
    if not use_conda:
        mlflow_run_arr.append("--no-conda")
    if run_id:
        mlflow_run_arr.extend(["--run-id", run_id])
    if parameters:
        for key, value in parameters.items():
            mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    final_env = os.environ.copy()
    final_env.update(env_map)
    if stream_output:
        popen = subprocess.Popen(
            mlflow_run_arr, cwd=work_dir, env=final_env, universal_newlines=True)
    else:
        popen = subprocess.Popen(
            mlflow_run_arr, cwd=work_dir, env=final_env, universal_newlines=True,
            stderr=open(os.devnull, "w"), stdout=open(os.devnull, "w"))
    return LocalSubmittedRun(run_id, popen, command)


def _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir,
                 experiment_id, run_id, block):
    """Locally run a project that has been checked out in `work_dir`."""
    # Try to build the command first in case the user mis-specified parameters
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    # Synchronously create the conda env before attempting to run the project to mitigate the risk
    # of failures due to concurrent attempts to create the same conda environment.
    if use_conda:
        conda_env_path = os.path.abspath(os.path.join(work_dir, project.conda_env))
        _maybe_create_conda_env(conda_env_path)

    # Create a new run and log every provided parameter into it (or get an existing run if run_id
    # is specified).
    if run_id:
        active_run = tracking._get_existing_run(run_id)
    else:
        active_run = tracking._create_run(
            experiment_id=experiment_id, source_name=project.uri,
            source_version=tracking._get_git_commit(work_dir), entry_point_name=entry_point,
            source_type=SourceType.PROJECT)
        if parameters is not None:
            for key, value in parameters.items():
                active_run.log_param(Param(key, value))

    # Add the run id into a magic environment variable that the subprocess will read,
    # causing it to reuse the run.
    env_map = {
        tracking._RUN_ID_ENV_VAR: active_run.run_info.run_uuid,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(experiment_id),
    }

    return _launch_local_run(
        project.uri, active_run.run_info.run_uuid, entry_point, parameters, work_dir, env_map,
        use_conda, storage_dir, stream_output=block)
