"""APIs for running MLflow Projects locally or remotely."""

from __future__ import print_function

import hashlib
import json
import os
import re
import shutil
import tempfile

from distutils import dir_util


from mlflow.projects._project_spec import Project
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
import mlflow.tracking as tracking
from mlflow.projects.submitted_run import SubmittedRun


from mlflow.utils import file_utils, process
from mlflow.utils.logging_utils import eprint

# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")


class ExecutionException(Exception):
    """Exception thrown when executing a project fails."""
    pass


def _run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
         mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
         use_temp_cwd=False, storage_dir=None, block=True):
    exp_id = experiment_id or tracking._get_experiment_id()
    if mode is None or mode == "local":
        return _run_local(
            uri=uri, entry_point=entry_point, version=version, parameters=parameters,
            experiment_id=exp_id, use_conda=use_conda, use_temp_cwd=use_temp_cwd,
            storage_dir=storage_dir, git_username=git_username, git_password=git_password,
            block=block)
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
        mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
        use_temp_cwd=False, storage_dir=None, block=True):
    """
    Run an MLflow project from the given URI.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.

    Raises:
      `mlflow.projects.ExecutionException` if a run launched in blocking mode is unsuccessful.

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
    :return: A `SubmittedRun` exposing information (e.g. run ID) about the launched run.
    """
    submitted_run_obj = _run(uri=uri, entry_point=entry_point, version=version,
                             parameters=parameters,
                             experiment_id=experiment_id,
                             mode=mode, cluster_spec=cluster_spec, git_username=git_username,
                             git_password=git_password, use_conda=use_conda,
                             use_temp_cwd=use_temp_cwd, storage_dir=storage_dir, block=block)
    if block:
        submitted_run_obj.wait()
        run_status = submitted_run_obj.get_status()
        if run_status and RunStatus.from_string(run_status) != RunStatus.FINISHED:
            raise ExecutionException("=== Run %s was unsuccessful, status: '%s' ===" %
                                     (submitted_run_obj.run_id, run_status))
    return submitted_run_obj


def _run_local(uri, entry_point, version, parameters, experiment_id, use_conda, use_temp_cwd,
               storage_dir, git_username, git_password, block):
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
    project = Project(expanded_uri, file_utils.read_yaml(work_dir, "MLproject"))
    return _run_project(
        project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id, block)


def _get_work_dir(uri, use_temp_cwd):
    """
    Returns a working directory to use for fetching & running the project with the specified URI.
    :param use_temp_cwd: Only used if `uri` is a local directory. If True, returns a temporary
                         working directory.
    """
    if _GIT_URI_REGEX.match(uri) or use_temp_cwd:
        # Create a temp directory to download and run the project in
        return tempfile.mkdtemp(prefix="mlflow-")
    return os.path.abspath(uri)


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _expand_uri(uri):
    if _GIT_URI_REGEX.match(uri):
        return uri
    return os.path.abspath(uri)


def _fetch_project(uri, version, dst_dir, git_username, git_password):
    """Download a project to the target `dst_dir` from a Git URI or local path."""
    if _GIT_URI_REGEX.match(uri):
        # Use Git to clone the project
        _fetch_git_repo(uri, version, dst_dir, git_username, git_password)
    else:
        if version is not None:
            raise ExecutionException("Setting a version is only supported for Git project URIs")
        # TODO: don't copy mlruns directory here
        # Note: uri might be equal to dst_dir, e.g. if we're not using a temporary work dir
        if uri != dst_dir:
            dir_util.copy_tree(src=uri, dst=dst_dir)

    # Make sure they don't have an outputs or mlruns directory (will need to change if we change
    # how we log results locally)
    shutil.rmtree(os.path.join(dst_dir, "outputs"), ignore_errors=True)
    shutil.rmtree(os.path.join(dst_dir, "mlruns"), ignore_errors=True)


def _fetch_git_repo(uri, version, dst_dir, git_username, git_password):
    """
    Clones the git repo at `uri` into `dst_dir`, checking out commit `version` (or defaulting
    to the head commit of the repository's master branch if version is unspecified). If git_username
    and git_password are specified, uses them to authenticate while fetching the repo. Otherwise,
    assumes authentication parameters are specified by the environment, e.g. by a Git credential
    helper.
    """
    # We defer importing git until the last moment, because the import requires that the git
    # executable is availble on the PATH, so we only want to fail if we actually need it.
    import git
    repo = git.Repo.init(dst_dir)
    origin = repo.create_remote("origin", uri)
    git_args = [git_username, git_password]
    if not (all(arg is not None for arg in git_args) or all(arg is None for arg in git_args)):
        raise ExecutionException("Either both or neither of git_username and git_password must be "
                                 "specified.")
    if git_username:
        git_credentials = "url=%s\nusername=%s\npassword=%s" % (uri, git_username, git_password)
        repo.git.config("--local", "credential.helper", "cache")
        process.exec_cmd(cmd=["git", "credential-cache", "store"], cwd=dst_dir,
                         cmd_stdin=git_credentials)
    origin.fetch()
    if version is not None:
        repo.git.checkout(version)
    else:
        repo.create_head("master", origin.refs.master)
        repo.heads.master.checkout()


def _get_conda_env_name(conda_env_path):
    with open(conda_env_path) as conda_env_file:
        conda_env_hash = hashlib.sha1(conda_env_file.read().encode("utf-8")).hexdigest()
    return "mlflow-%s" % conda_env_hash


def _maybe_create_conda_env(conda_env_path):
    conda_env = _get_conda_env_name(conda_env_path)
    try:
        process.exec_cmd(["conda", "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException('conda is not installed properly. Please follow the instructions '
                                 'on https://conda.io/docs/user-guide/install/index.html')
    (_, stdout, _) = process.exec_cmd(["conda", "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]

    conda_action = 'create'
    if conda_env not in env_names:
        eprint('=== Creating conda environment %s ===' % conda_env)
        process.exec_cmd(["conda", "env", conda_action, "-n", conda_env, "--file",
                          conda_env_path], stream_output=True)


def _launch_local_run(active_run, command, work_dir, env_map, stream_output):
    """
    Runs an entry point by launching its command in a subprocess, updating the tracking server with
    the run's exit status.

    :param active_run: `ActiveRun` to which to post status updates for the launched run
    :param command: Entry point command to execute
    :param work_dir: Working directory to use when executing `command`
    :param env_map: Dict of environment variable key-value pairs to set in the process for `command`
    :return `SubmittedRun` corresponding to the launched run.
    """
    from mlflow.projects.pollable_run import LocalPollableRun
    pollable_run = LocalPollableRun(
        command=command, work_dir=work_dir, env_map=env_map, stream_output=stream_output)
    return SubmittedRun(active_run, pollable_run)


def _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir,
                 experiment_id, block):
    """Locally run a project that has been checked out in `work_dir`."""
    storage_dir_for_run = _get_storage_dir(storage_dir)
    eprint("=== Created directory %s for downloading remote URIs passed to arguments of "
           "type 'path' ===" % storage_dir_for_run)
    # Try to build the command first in case the user mis-specified parameters
    run_project_command = project.get_entry_point(entry_point)\
        .compute_command(parameters, storage_dir_for_run)
    commands = []
    if use_conda:
        conda_env_path = os.path.abspath(os.path.join(work_dir, project.conda_env))
        _maybe_create_conda_env(conda_env_path)
        commands.append("source activate %s" % _get_conda_env_name(conda_env_path))

    # Create a new run and log every provided parameter into it.
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

    commands.append(run_project_command)
    command = " && ".join(commands)
    eprint("=== Running command '%s' in run with ID '%s' === "
           % (command, active_run.run_info.run_uuid))

    return _launch_local_run(
        active_run, command, work_dir, env_map, stream_output=block)
