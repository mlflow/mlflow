"""APIs for running MLflow Projects locally or remotely."""

from __future__ import print_function

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import time

from distutils import dir_util

from six.moves import shlex_quote
from databricks_cli.configure import provider

from mlflow.projects._project_spec import Project
from mlflow.version import VERSION
from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
import mlflow.tracking as tracking
from mlflow.tracking.runs import SubmittedRun


from mlflow.utils import file_utils, process, rest_utils
from mlflow.utils.logging_utils import eprint


# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")


class ExecutionException(Exception):
    """Exception thrown when executing a project fails."""
    pass


def run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
        mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
        use_temp_cwd=False, storage_dir=None, block=True):
    """
    Run an MLflow project from the given URI.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.

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
    :return: A `SubmittedRun` exposing information (e.g. run ID) about the launched run.
    """
    if mode is None or mode == "local":
        return _run_local(uri=uri, entry_point=entry_point, version=version, parameters=parameters,
                          experiment_id=experiment_id, use_conda=use_conda, use_temp_cwd=use_temp_cwd,
                          storage_dir=storage_dir, git_username=git_username, git_password=git_password,
                          block=block)
    elif mode == "databricks":
        return _run_databricks(uri=uri, entry_point=entry_point, version=version,
                               parameters=parameters, experiment_id=experiment_id, cluster_spec=cluster_spec,
                               git_username=git_username, git_password=git_password, block=block)
    else:
        supported_modes = ["local", "databricks"]
        raise ExecutionException("Got unsupported execution mode %s. Supported "
                                 "values: %s" % (mode, supported_modes))


def _get_databricks_run_cmd(uri, entry_point, version, parameters):
    """
    Generates MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks
    """
    mlflow_run_cmd = ["mlflow", "run", uri, "--entry-point", entry_point]
    if version is not None:
        mlflow_run_cmd.extend(["--version", version])
    if parameters is not None:
        for key, value in parameters.items():
            mlflow_run_cmd.extend(["-P", "%s=%s" % (key, value)])
    mlflow_run_str = " ".join(map(shlex_quote, mlflow_run_cmd))
    return ["bash", "-c", "export PATH=$PATH:$DB_HOME/python/bin:/$DB_HOME/conda/bin && %s"
            % mlflow_run_str]


def _get_databricks_hostname_and_auth():
    """
    Reads the hostname & auth token to use for running on Databricks from the config file created
    by the Databricks CLI.
    """
    home_dir = os.path.expanduser("~")
    cfg_file = os.path.join(home_dir, ".databrickscfg")
    if not os.path.exists(cfg_file):
        raise ExecutionException("Could not find profile for Databricks CLI in %s. Make sure the "
                                 "the Databricks CLI is installed and that credentials have been "
                                 "configured as described in "
                                 "https://github.com/databricks/databricks-cli" % cfg_file)
    else:
        config = provider.get_config_for_profile(provider.DEFAULT_SECTION)
        return config.host, config.token, config.username, config.password


def _do_databricks_run(project_uri, command, env_vars, cluster_spec):
    """
    Runs the specified shell command on a Databricks cluster.
    :param project_uri: URI of the project from which our shell command originates
    :param command: Shell command to run
    :param env_vars: Environment variables to set in the process running `command`
    :param cluster_spec: Dictionary describing the cluster, expected to contain the fields for a
                         NewCluster (see
                         https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster)
    :return: The ID of the Databricks Job Run. Can be used to query the run's status via the
             Databricks Runs Get API (https://docs.databricks.com/api/latest/jobs.html#runs-get).
    """
    hostname, token, username, password, = _get_databricks_hostname_and_auth()
    auth = (username, password) if username is not None and password is not None else None
    # Make jobs API request to launch run.
    req_body_json = {
        'run_name': 'MLflow Run for %s' % project_uri,
        'new_cluster': cluster_spec,
        'shell_command_task': {
            'command': command,
            "env_vars": env_vars
        },
        "libraries": [{"pypi": {"package": "mlflow==%s" % VERSION}}]
    }
    run_submit_res = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/submit", token=token, auth=auth, method="POST",
        req_body_json=req_body_json)
    databricks_run_id = run_submit_res["run_id"]
    eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
           "page URL... ===" % databricks_run_id)
    run_info = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/get", token=token, auth=auth, method="GET",
        params={"run_id": databricks_run_id})
    jobs_page_url = run_info["run_page_url"]
    eprint("=== Check the run's status at %s ===" % jobs_page_url)
    return databricks_run_id


def _create_databricks_run(experiment_id, source_name, source_version, entry_point_name):
    """
    Makes an API request to the current tracking server to create a new run with the specified
    attributes. Returns an `ActiveRun` that can be used to query the tracking server for the run's
    status or log metrics/params for the run.
    """
    # Figure out tracking URI
    tracking_uri = tracking.get_tracking_uri()
    if tracking.is_local_uri(tracking_uri):
        # TODO: we'll actually use the Databricks deployment's tracking URI here in the future
        eprint("WARNING: MLflow tracking URI is set to a local URI (%s), so results from Databricks"
               "will not be logged permanently." % tracking_uri)
        return None
    else:
        # Assume non-local tracking URIs are accessible from Databricks (won't work for e.g.
        # localhost)
        return tracking._create_run(experiment_id=experiment_id,
                                    source_name=source_name,
                                    source_version=source_version,
                                    entry_point_name=entry_point_name,
                                    source_type=SourceType.PROJECT)


def _get_databricks_run_result_status(databricks_run_id):
    """
    Returns the run result status (string) of the Databricks run with the passed-in ID, or None
    if the run is still active. See possible values at
    https://docs.databricks.com/api/latest/jobs.html#runresultstate.
    """
    hostname, token, username, password, = _get_databricks_hostname_and_auth()
    auth = (username, password) if username is not None and password is not None else None
    res = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/get", token=token, auth=auth, method="GET",
        params={"run_id": databricks_run_id})
    return res["state"].get("result_state", None)


def _wait_databricks(databricks_run_id, sleep_interval=30):
    """
    Polls a Databricks Job run (with run ID `databricks_run_id`) for termination, checking the
    run's status every `sleep_interval` seconds.
    """
    result_state = None
    while result_state is None:
        result_state = _get_databricks_run_result_status(databricks_run_id)
        eprint("=== Databricks run is still active, checking run status again after %s seconds "
               "===" % sleep_interval)
        time.sleep(sleep_interval)
    if result_state != "SUCCESS":
        raise ExecutionException("=== Databricks run finished with status %s != 'SUCCESS' "
                                 "===" % result_state)
    eprint("=== Run succeeded ===")


def _run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec,
                    git_username, git_password, block):
    # Create run object with remote tracking server
    remote_run = _create_databricks_run(experiment_id, source_name=uri, source_version=version,
                                        entry_point_name=entry_point)
    # Set up environment variables for remote execution
    env_vars = {"MLFLOW_GIT_URI": uri}
    if git_username is not None:
        env_vars["MLFLOW_GIT_USERNAME"] = git_username
    if git_password is not None:
        env_vars["MLFLOW_GIT_PASSWORD"] = git_password
    if experiment_id is not None:
        eprint("=== Using experiment ID %s ===" % experiment_id)
        env_vars[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    if remote_run is not None:
        env_vars[tracking._TRACKING_URI_ENV_VAR] = tracking.get_tracking_uri()
        env_vars[tracking._RUN_ID_ENV_VAR] = remote_run.run_info.run_uuid
    eprint("=== Running entry point %s of project %s on Databricks. ===" % (entry_point, uri))
    # Launch run on Databricks
    with open(cluster_spec, 'r') as handle:
        cluster_spec = json.load(handle)
    command = _get_databricks_run_cmd(uri, entry_point, version, parameters)
    db_run_id = _do_databricks_run(uri, command, env_vars, cluster_spec)
    if block:
        eprint("=== Waiting for Databricks Job Run to complete ===")
        _wait_databricks(db_run_id)
    return SubmittedRun(None) if remote_run is None else SubmittedRun(remote_run.run_info.run_uuid)


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


def _launch_local_command(active_run, command, work_dir, env_map):
    try:
        process.exec_cmd([os.environ.get("SHELL", "bash"), "-c", command], cwd=work_dir,
                         stream_output=True, env=env_map)
        eprint("=== Run succeeded ===")
        active_run.set_terminated("FINISHED")
    except process.ShellCommandException:
        active_run.set_terminated("FAILED")
        eprint("=== Run failed ===")


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
    exp_id_for_run = experiment_id or tracking._get_experiment_id()
    assert tracking._active_run is None
    active_run = tracking._create_run(
        experiment_id=exp_id_for_run, source_name=project.uri,
        source_version=tracking._get_git_commit(work_dir), entry_point_name=entry_point,
        source_type=SourceType.PROJECT)
    assert tracking._active_run is None
    if parameters is not None:
        for key, value in parameters.items():
            active_run.log_param(Param(key, value))
    # Add the run id into a magic environment variable that the subprocess will read,
    # causing it to reuse the run.
    exp_id = experiment_id or tracking._get_experiment_id()
    env_map = {
        tracking._RUN_ID_ENV_VAR: active_run.run_info.run_uuid,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(exp_id),
    }

    commands.append(run_project_command)
    command = " && ".join(commands)
    eprint("=== Running command: %s ===" % command)

    if block:
        _launch_local_command(active_run, command, work_dir, env_map)
    else:
        # Launch monitoring process that launches a subprocess for the run & posts the run's status
        # to the tracking server.
        process = multiprocessing.Process(
            target=_launch_local_command, args=(active_run, command, work_dir, env_map))
        process.start()
    return SubmittedRun(active_run.run_info.run_uuid)
