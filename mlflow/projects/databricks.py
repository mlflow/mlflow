import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time
from databricks_cli.configure import provider
from six.moves import shlex_quote, urllib

from mlflow.entities.experiment import Experiment
from mlflow.entities.source_type import SourceType

from mlflow.projects import ExecutionException, _fetch_project, _get_work_dir, _load_project
from mlflow.projects.pollable_run import DatabricksPollableRun
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.utils import rest_utils, file_utils, process
from mlflow.utils.logging_utils import eprint
from mlflow import tracking
from mlflow.version import VERSION

DB_CONTAINER_BASE = "/databricks/mlflow"
DB_TARFILE_BASE = os.path.join(DB_CONTAINER_BASE, "project-tars")
DB_PROJECTS_BASE = os.path.join(DB_CONTAINER_BASE, "projects")
DB_TARFILE_ARCHIVE_NAME = "mlflow-project"
DBFS_EXPERIMENT_DIR_BASE = "mlflow-experiments"


def _jobs_runs_get(databricks_run_id, profile):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/get", method="GET", params={"run_id": databricks_run_id},
        profile=profile)


def _jobs_runs_cancel(databricks_run_id, profile):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/cancel", method="POST", req_body_json={"run_id": databricks_run_id},
        profile=profile)


def _jobs_runs_submit(req_body_json, profile):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/submit", method="POST", req_body_json=req_body_json, profile=profile)


def _get_databricks_run_cmd(dbfs_fuse_tar_uri, entry_point, parameters):
    """
    Generates MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks
    """
    # Strip ".gz" and ".tar" file extensions from base filename of the tarfile
    tar_hash = os.path.splitext(os.path.splitext(os.path.basename(dbfs_fuse_tar_uri))[0])[0]
    container_tar_path = os.path.abspath(os.path.join(DB_TARFILE_BASE,
                                                      os.path.basename(dbfs_fuse_tar_uri)))
    project_dir = os.path.join(DB_PROJECTS_BASE, tar_hash)
    mlflow_run_arr = list(map(shlex_quote, ["mlflow", "run", project_dir, "--new-dir",
                                            "--entry-point", entry_point]))
    if parameters is not None:
        for key, value in parameters.items():
            mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    mlflow_run_cmd = " ".join(mlflow_run_arr)
    shell_command = textwrap.dedent("""
    export PATH=$PATH:$DB_HOME/python/bin:/$DB_HOME/conda/bin &&
    mlflow --version &&
    # Make local directories in the container into which to copy/extract the tarred project
    mkdir -p {0} && mkdir -p {1} &&
    # Rsync from DBFS FUSE to avoid copying archive into local filesystem if it already exists
    rsync -a -v --ignore-existing {2} {0} &&
    # Extract project into a temporary directory. We don't extract directly into the desired
    # directory as tar extraction isn't guaranteed to be atomic
    cd $(mktemp -d) &&
    tar -xzvf {3} &&
    # Atomically move the extracted project into the desired directory
    mv -T {4} {5} &&
    {6}
    """.format(DB_TARFILE_BASE, DB_PROJECTS_BASE, dbfs_fuse_tar_uri, container_tar_path,
               DB_TARFILE_ARCHIVE_NAME, project_dir, mlflow_run_cmd))
    return ["bash", "-c", shell_command]


def _check_databricks_cli_configured():
    cfg_file = os.path.join(os.path.expanduser("~"), ".databrickscfg")
    try:
        process.exec_cmd(["databricks", "--version"])
    except process.ShellCommandException:
        raise ExecutionException(
            "Could not find Databricks CLI on PATH. Please install and configure the Databricks "
            "CLI as described in https://github.com/databricks/databricks-cli")
    if not os.path.exists(cfg_file):
        raise ExecutionException("Could not find profile for Databricks CLI in %s. Make sure the "
                                 "the Databricks CLI is installed and that credentials have been "
                                 "configured as described in "
                                 "https://github.com/databricks/databricks-cli" % cfg_file)


def _upload_to_dbfs(src_path, dbfs_uri, profile):
    """
    Uploads the file at `src_path` to the specified DBFS URI within the Databricks workspace
    corresponding to the passed-in Databricks CLI profile.
    """
    process.exec_cmd(cmd=["databricks", "fs", "cp", src_path, dbfs_uri,
                          "--profile", profile])


def _dbfs_path_exists(dbfs_uri, profile):
    """
    Returns True if the passed-in path exists in DBFS for the workspace corresponding to the
    passed-in Databricks CLI profile.
    """
    dbfs_path = _parse_dbfs_uri_path(dbfs_uri)
    try:
        res = rest_utils.databricks_api_request(
            endpoint="dbfs/get-status", method="GET", params={"path": dbfs_path}, profile=profile)
        return res
    # Assume that CLI command failure -> the file didn't exist
    except process.ShellCommandException:
        return False


def _upload_project_to_dbfs(project_dir, experiment_id, profile):
    """
    Tars a project directory into an archive in a temp dir, returning the path to the
    tarball.
    """
    temp_tarfile_dir = tempfile.mkdtemp()
    temp_tar_filename = file_utils.build_path(temp_tarfile_dir, "project.tar.gz")
    try:
        file_utils.make_tarfile(temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME)
        commit = tracking._get_git_commit(project_dir)
        if commit is not None:
            tarfile_name = os.path.join("git-projects", commit)
        else:
            with open(temp_tar_filename, "rb") as tarred_project:
                tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
            tarfile_name = os.path.join("local-projects", tarfile_hash)
        # TODO: Get subdirectory for experiment from the tracking server
        dbfs_uri = os.path.join("dbfs:/", DBFS_EXPERIMENT_DIR_BASE, str(experiment_id),
                                 "%s.tar.gz" % tarfile_name)
        eprint("=== Uploading project to DBFS path %s ===" % dbfs_uri)
        if not _dbfs_path_exists(dbfs_uri, profile):
            _upload_to_dbfs(temp_tar_filename, dbfs_uri, profile)
        else:
            eprint("=== Project already exists in DBFS ===")
        eprint("=== Finished uploading project to %s ===" % dbfs_uri)
    finally:
        shutil.rmtree(temp_tarfile_dir)
    return dbfs_uri


def _get_run_result_state(databricks_run_id, profile):
    """
    Returns the run result state (string) of the Databricks run with the passed-in ID, or None
    if the run is still active. See possible values at
    https://docs.databricks.com/api/latest/jobs.html#runresultstate.
    """
    res = _jobs_runs_get(databricks_run_id, profile)
    return res["state"].get("result_state", None)


def _run_shell_command_job(project_uri, command, env_vars, cluster_spec, profile):
    """
    Runs the specified shell command on a Databricks cluster.
    :param project_uri: URI of the project from which our shell command originates
    :param command: Shell command to run
    :param env_vars: Environment variables to set in the process running `command`
    :param cluster_spec: Dictionary describing the cluster, expected to contain the fields for a
                         NewCluster (see
                         https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster)
    :param profile: Databricks CLI profile describing the workspace/auth to use when making
                    Databricks API requests.
    :return: The ID of the Databricks Job Run. Can be used to query the run's status via the
             Databricks Runs Get API (https://docs.databricks.com/api/latest/jobs.html#runs-get).
    """
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
    run_submit_res = _jobs_runs_submit(req_body_json, profile)
    databricks_run_id = run_submit_res["run_id"]
    eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
           "page URL... ===" % databricks_run_id)
    run_info = _jobs_runs_get(databricks_run_id, profile)
    jobs_page_url = run_info["run_page_url"]
    eprint("=== Check the run's status at %s ===" % jobs_page_url)
    return databricks_run_id


def _create_databricks_run(tracking_uri, experiment_id, source_name, source_version,
                           entry_point_name):
    """
    Makes an API request to the specified tracking server to create a new run with the specified
    attributes. Returns an `ActiveRun` that can be used to query the tracking server for the run's
    status or log metrics/params for the run.
    """
    if tracking.is_local_uri(tracking_uri):
        # TODO: we'll actually use the Databricks deployment's tracking URI here in the future
        eprint("WARNING: MLflow tracking URI is set to a local URI (%s), so results from "
               "Databricks will not be logged permanently." % tracking_uri)
        return None
    else:
        # Assume non-local tracking URIs are accessible from Databricks (won't work for e.g.
        # localhost)
        return tracking._create_run(experiment_id=experiment_id,
                                    source_name=source_name,
                                    source_version=source_version,
                                    entry_point_name=entry_point_name,
                                    source_type=SourceType.PROJECT)


def _parse_dbfs_uri_path(dbfs_uri):
    """Parses and returns the absolute path within DBFS of the file with the specified URI"""
    return urllib.parse.urlparse(dbfs_uri).path


def run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec, db_profile,
                   git_username, git_password):
    """
    Runs a project on Databricks, returning a `SubmittedRun` that can be used to query the run's
    status or wait for the resulting Databricks Job run to terminate.
    """
    _check_databricks_cli_configured()
    if cluster_spec is None:
        raise ExecutionException("Cluster spec must be provided when launching MLflow project runs "
                                 "on Databricks.")
    databricks_profile = db_profile or provider.DEFAULT_SECTION

    # Fetch the project into work_dir & validate parameters
    work_dir = _get_work_dir(uri, use_temp_cwd=False)
    _fetch_project(uri, version, work_dir, git_username, git_password)
    project = _load_project(work_dir, uri)
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    # Upload the project to DBFS, get the URI of the project
    final_experiment_id = experiment_id or Experiment.DEFAULT_EXPERIMENT_ID
    dbfs_project_uri = _upload_project_to_dbfs(work_dir, final_experiment_id, databricks_profile)

    # Create run object with remote tracking server. Get the git commit from the working directory,
    # etc.
    tracking_uri = tracking.get_tracking_uri()
    remote_run = _create_databricks_run(
        tracking_uri=tracking_uri, experiment_id=experiment_id, source_name=project.uri,
        source_version=tracking._get_git_commit(work_dir), entry_point_name=entry_point)
    # Set up environment variables for remote execution
    env_vars = {}
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
    fuse_dst_dir = os.path.join("/dbfs/", _parse_dbfs_uri_path(dbfs_project_uri))
    command = _get_databricks_run_cmd(fuse_dst_dir, entry_point, parameters)
    db_run_id = _run_shell_command_job(uri, command, env_vars, cluster_spec, databricks_profile)
    return SubmittedRun(remote_run, DatabricksPollableRun(db_run_id, databricks_profile))


def cancel_databricks(databricks_run_id, profile):
    _jobs_runs_cancel(databricks_run_id, profile)


def monitor_databricks(databricks_run_id, profile, sleep_interval=30):
    """
    Polls a Databricks Job run (with run ID `databricks_run_id`) for termination, checking the
    run's status every `sleep_interval` seconds.
    """
    result_state = _get_run_result_state(databricks_run_id, profile)
    while result_state is None:
        time.sleep(sleep_interval)
        result_state = _get_run_result_state(databricks_run_id, profile)
    return result_state == "SUCCESS"
