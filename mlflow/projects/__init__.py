"""APIs for running MLflow projects locally or remotely."""

from __future__ import print_function

from distutils import dir_util
import hashlib
import json
import os
import re
import subprocess
import tempfile

from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow.projects._project_spec import Project
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
import mlflow.tracking as tracking


from mlflow.utils import file_utils, process
from mlflow.utils.logging_utils import eprint

# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")
# Environment variable indicating a path to a conda installation. MLflow will default to running
# "conda" if unset
MLFLOW_CONDA_HOME = "MLFLOW_CONDA_HOME"


class ExecutionException(Exception):
    """Exception thrown when executing a project fails."""
    pass


def _run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
         mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
         storage_dir=None, block=True, run_id=None):
    """
    Helper that delegates to the project-running method corresponding to the passed-in mode.
    Returns a ``SubmittedRun`` corresponding to the project run.
    """
    exp_id = experiment_id or tracking._get_experiment_id()
    parameters = parameters or {}
    if mode == "databricks":
        from mlflow.projects.databricks import run_databricks
        return run_databricks(
            uri=uri, entry_point=entry_point, version=version, parameters=parameters,
            experiment_id=exp_id, cluster_spec=cluster_spec, git_username=git_username,
            git_password=git_password)
    elif mode == "local" or mode is None:
        work_dir = _fetch_project(uri=uri, force_tempdir=False, version=version,
                                  git_username=git_username, git_password=git_password)
        project = _load_project(project_dir=work_dir)
        project.get_entry_point(entry_point)._validate_parameters(parameters)
        # Synchronously create a conda environment (even though this may take some time) to avoid
        # failures due to multiple concurrent attempts to create the same conda env.
        if use_conda:
            _maybe_create_conda_env(conda_env_path=os.path.join(work_dir, project.conda_env))
        if run_id:
            active_run = tracking._get_existing_run(run_id)
        else:
            active_run = _create_run(uri, exp_id, work_dir, entry_point, parameters)
        # In blocking mode, run the entry point command in blocking fashion, sending status updates
        # to the tracking server when finished. Note that the run state may not be persisted to the
        # tracking server if interrupted
        if block:
            command = _get_entry_point_command(
                work_dir, entry_point, use_conda, parameters, storage_dir)
            return _run_entry_point(command, work_dir, exp_id, run_id=active_run.run_info.run_uuid)
        # Otherwise, invoke `mlflow run` in a subprocess
        return _invoke_mlflow_run_subprocess(
            work_dir=work_dir, entry_point=entry_point, parameters=parameters, experiment_id=exp_id,
            use_conda=use_conda, storage_dir=storage_dir, run_id=active_run.run_info.run_uuid)
    supported_modes = ["local", "databricks"]
    raise ExecutionException("Got unsupported execution mode %s. Supported "
                             "values: %s" % (mode, supported_modes))


def run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
        mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
        storage_dir=None, block=True, run_id=None):
    """
    Run an MLflow project from the given URI.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the ``version``.

    :raises mlflow.projects.ExecutionException: if a run launched in blocking mode is unsuccessful.

    :param uri: URI of project to run. Expected to be either a relative/absolute local filesystem
                path or a git repository URI (e.g. https://github.com/mlflow/mlflow-example)
                pointing to a project directory containing an MLproject file.
    :param entry_point: Entry point to run within the project. If no entry point with the specified
                        name is found, attempts to run the project file ``entry_point`` as a script,
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
    :param storage_dir: Only used if ``mode`` is local. MLflow will download artifacts from
                        distributed URIs passed to parameters of type 'path' to subdirectories of
                        ``storage_dir``.
    :param block: Whether or not to block while waiting for a run to complete. Defaults to True.
                  Note that if ``block`` is False and mode is "local", this method will return, but
                  the current process will block when exiting until the local run completes.
                  If the current process is interrupted, any asynchronous runs launched via this
                  method will be terminated.
    :param run_id: Note: this argument is used internally by the MLflow project APIs and should
                   not be specified. If specified, the given run ID will be used instead of
                   creating a new run.
    :return: A ``SubmittedRun`` exposing information (e.g. run ID) about the launched run.
              The returned ``SubmittedRun`` is not thread-safe.
    """
    submitted_run_obj = _run(
        uri=uri, entry_point=entry_point, version=version, parameters=parameters,
        experiment_id=experiment_id, mode=mode, cluster_spec=cluster_spec,
        git_username=git_username, git_password=git_password, use_conda=use_conda,
        storage_dir=storage_dir, block=block, run_id=run_id)
    if block:
        _wait_for(submitted_run_obj)
    return submitted_run_obj


def _wait_for(submitted_run_obj):
    """Wait on the passed-in submitted run, reporting its status to the tracking server."""
    run_id = submitted_run_obj.run_id
    active_run = None
    # Note: there's a small chance we fail to report the run's status to the tracking server if
    # we're interrupted before we reach the try block below
    try:
        active_run = tracking._get_existing_run(run_id) if run_id is not None else None
        if submitted_run_obj.wait():
            eprint("=== Run (ID '%s') succeeded ===" % run_id)
            _maybe_set_run_terminated(active_run, "FINISHED")
        else:
            _maybe_set_run_terminated(active_run, "FAILED")
            raise ExecutionException("=== Run (ID '%s') failed ===" % run_id)
    except KeyboardInterrupt:
        eprint("=== Run (ID '%s') === interrupted, cancelling run ===" % run_id)
        submitted_run_obj.cancel()
        _maybe_set_run_terminated(active_run, "FAILED")
        raise


def _load_project(project_dir):
    return Project(file_utils.read_yaml(project_dir, "MLproject"))


def _parse_subdirectory(uri):
    # Parses a uri and returns the uri and subdirectory as separate values.
    # Uses '#' as a delimiter.
    subdirectory = ''
    parsed_uri = uri
    if '#' in uri:
        subdirectory = uri[uri.find('#')+1:]
        parsed_uri = uri[:uri.find('#')]
    if subdirectory and '.' in subdirectory:
        raise ExecutionException("'.' is not allowed in project subdirectory paths.")
    return parsed_uri, subdirectory


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _expand_uri(uri):
    if _is_local_uri(uri):
        return os.path.abspath(uri)
    return uri


def _is_local_uri(uri):
    """Returns True if the passed-in URI should be interpreted as a path on the local filesystem."""
    return not _GIT_URI_REGEX.match(uri)


def _fetch_project(uri, force_tempdir, version=None, git_username=None, git_password=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory.
    :param force_tempdir: If True, will fetch the project into a temporary directory. Otherwise,
                          will fetch Git projects into a temporary directory but simply return the
                          path of local projects (i.e. perform a no-op for local projects).
    """
    parsed_uri, subdirectory = _parse_subdirectory(uri)
    use_temp_dst_dir = force_tempdir or not _is_local_uri(parsed_uri)
    dst_dir = tempfile.mkdtemp() if use_temp_dst_dir else parsed_uri
    if use_temp_dst_dir:
        eprint("=== Fetching project from %s into %s ===" % (uri, dst_dir))
    if _is_local_uri(uri):
        if version is not None:
            raise ExecutionException("Setting a version is only supported for Git project URIs")
        if use_temp_dst_dir:
            dir_util.copy_tree(src=parsed_uri, dst=dst_dir)
    else:
        assert _GIT_URI_REGEX.match(parsed_uri), "Non-local URI %s should be a Git URI" % parsed_uri
        _fetch_git_repo(parsed_uri, version, dst_dir, git_username, git_password)

    # Make sure there is a MLproject file in the specified working directory.
    if not os.path.isfile(os.path.join(dst_dir, subdirectory, "MLproject")):
        if subdirectory == '':
            raise ExecutionException("No MLproject file found in %s" % uri)
        else:
            raise ExecutionException("No MLproject file found in subdirectory %s of %s" %
                                     (subdirectory, parsed_uri))

    return os.path.abspath(os.path.join(dst_dir, subdirectory))


def _fetch_git_repo(uri, version, dst_dir, git_username, git_password):
    """
    Clone the git repo at ``uri`` into ``dst_dir``, checking out commit ``version`` (or defaulting
    to the head commit of the repository's master branch if version is unspecified).
    If ``git_username`` and ``git_password`` are specified, uses them to authenticate while fetching
    the repo. Otherwise, assumes authentication parameters are specified by the environment,
    e.g. by a Git credential helper.
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


def _get_conda_bin_executable(executable_name):
    """
    Return path to the specified executable, assumed to be discoverable within the 'bin'
    subdirectory of a conda installation.

    The conda home directory (expected to contain a 'bin' subdirectory) is configurable via the
    ``mlflow.projects.MLFLOW_CONDA_HOME`` environment variable. If
    ``mlflow.projects.MLFLOW_CONDA_HOME`` is unspecified, this method simply returns the passed-in
    executable name.
    """
    conda_home = os.environ.get(MLFLOW_CONDA_HOME)
    if conda_home:
        return os.path.join(conda_home, "bin/%s" % executable_name)
    return executable_name


def _maybe_create_conda_env(conda_env_path):
    conda_env = _get_conda_env_name(conda_env_path)
    conda_path = _get_conda_bin_executable("conda")
    try:
        process.exec_cmd([conda_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException("Could not find Conda executable at {0}. "
                                 "Ensure Conda is installed as per the instructions "
                                 "at https://conda.io/docs/user-guide/install/index.html. You can "
                                 "also configure MLflow to look for a specific Conda executable "
                                 "by setting the {1} environment variable to the path of the Conda "
                                 "executable".format(conda_path, MLFLOW_CONDA_HOME))
    (_, stdout, _) = process.exec_cmd([conda_path, "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]

    conda_action = 'create'
    if conda_env not in env_names:
        eprint('=== Creating conda environment %s ===' % conda_env)
        process.exec_cmd([conda_path, "env", conda_action, "-n", conda_env, "--file",
                          conda_env_path], stream_output=True)


def _maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run and not RunStatus.is_terminated(active_run.get_run().info.status):
        active_run.set_terminated(status)


def _get_entry_point_command(project_dir, entry_point, use_conda, parameters, storage_dir):
    project = _load_project(project_dir=project_dir)
    storage_dir_for_run = _get_storage_dir(storage_dir)
    eprint("=== Created directory %s for downloading remote URIs passed to arguments of "
           "type 'path' ===" % storage_dir_for_run)
    commands = []
    if use_conda:
        conda_env_path = os.path.abspath(os.path.join(project_dir, project.conda_env))
        activate_path = _get_conda_bin_executable("activate")
        commands.append("source %s %s" % (activate_path, _get_conda_env_name(conda_env_path)))
    commands.append(
        project.get_entry_point(entry_point).compute_command(parameters, storage_dir_for_run))
    return " && ".join(commands)


def _run_entry_point(command, work_dir, experiment_id, run_id):
    """
    Run an entry point command in a subprocess, returning a SubmittedRun that can be used to
    query the run's status.
    :param command: Entry point command to run
    :param work_dir: Working directory in which to run the command
    :param run_id: MLflow run ID associated with the entry point execution.
    """
    env = os.environ.copy()
    env.update(_get_run_env_vars(run_id, experiment_id))
    eprint("=== Running command '%s' in run with ID '%s' === " % (command, run_id))
    process = subprocess.Popen(["bash", "-c", command], close_fds=True, cwd=work_dir, env=env)
    return LocalSubmittedRun(run_id, process)


def _build_mlflow_run_cmd(
        uri, entry_point, storage_dir, use_conda, run_id, parameters):
    """
    Build and return an array containing an ``mlflow run`` command that can be invoked to locally
    run the project at the specified URI.
    """
    mlflow_run_arr = ["mlflow", "run", uri, "-e", entry_point, "--run-id", run_id]
    if storage_dir is not None:
        mlflow_run_arr.extend(["--storage-dir", storage_dir])
    if not use_conda:
        mlflow_run_arr.append("--no-conda")
    for key, value in parameters.items():
        mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    return mlflow_run_arr


def _run_mlflow_run_cmd(mlflow_run_arr, env_map):
    """
    Invoke ``mlflow run`` in a subprocess, which in turn runs the entry point in a child process.
    Returns a handle to the subprocess. Popen launched to invoke ``mlflow run``.
    """
    final_env = os.environ.copy()
    final_env.update(env_map)
    # Launch `mlflow run` command as the leader of its own process group so that we can do a
    # best-effort cleanup of all its descendant processes if needed
    return subprocess.Popen(
        mlflow_run_arr, env=final_env, universal_newlines=True, preexec_fn=os.setsid)


def _create_run(uri, experiment_id, work_dir, entry_point, parameters):
    """
    Create a ``Run`` against the current MLflow tracking server, logging metadata (e.g. the URI,
    entry point, and parameters of the project) about the run. Return an ``ActiveRun`` that can be
    used to report additional data about the run (metrics/params) to the tracking server.
    """
    active_run = tracking._create_run(
        experiment_id=experiment_id, source_name=_expand_uri(uri),
        source_version=tracking._get_git_commit(work_dir), entry_point_name=entry_point,
        source_type=SourceType.PROJECT)
    if parameters is not None:
        for key, value in parameters.items():
            active_run.log_param(Param(key, value))
    return active_run


def _get_run_env_vars(run_id, experiment_id):
    """
    Returns a dictionary of environment variable key-value pairs to set in subprocess launched
    to run MLflow projects.
    """
    return {
        tracking._RUN_ID_ENV_VAR: run_id,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(experiment_id),
    }


def _invoke_mlflow_run_subprocess(
        work_dir, entry_point, parameters, experiment_id, use_conda, storage_dir, run_id):
    """
    Run an MLflow project asynchronously by invoking ``mlflow run`` in a subprocess, returning
    a SubmittedRun that can be used to query run status.
    """
    eprint("=== Asynchronously launching MLflow run with ID %s ===" % run_id)
    mlflow_run_arr = _build_mlflow_run_cmd(
        uri=work_dir, entry_point=entry_point, storage_dir=storage_dir, use_conda=use_conda,
        run_id=run_id, parameters=parameters)
    mlflow_run_subprocess = _run_mlflow_run_cmd(
        mlflow_run_arr, _get_run_env_vars(run_id, experiment_id))
    return LocalSubmittedRun(run_id, mlflow_run_subprocess)
