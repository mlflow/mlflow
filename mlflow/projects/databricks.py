import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time

from six.moves import shlex_quote, urllib

from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType


from mlflow.projects import ExecutionException, _fetch_project, _load_project, _expand_uri
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.utils import rest_utils, file_utils, process
from mlflow.utils.logging_utils import eprint
from mlflow import tracking
from mlflow.version import VERSION

# Base directory within driver container for storing files related to MLflow
DB_CONTAINER_BASE = "/databricks/mlflow"
# Base directory within driver container for storing project archives
DB_TARFILE_BASE = os.path.join(DB_CONTAINER_BASE, "project-tars")
# Base directory directory within driver container for storing extracted project directories
DB_PROJECTS_BASE = os.path.join(DB_CONTAINER_BASE, "projects")
# Name to use for project directory when archiving it for upload to DBFS; the TAR will contain
# a single directory with this name
DB_TARFILE_ARCHIVE_NAME = "mlflow-project"
# Base directory within DBFS for storing code for project runs for experiments
DBFS_EXPERIMENT_DIR_BASE = "mlflow-experiments"


def _jobs_runs_get(databricks_run_id):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/get", method="GET", json={"run_id": databricks_run_id})


def _jobs_runs_cancel(databricks_run_id):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/cancel", method="POST", json={"run_id": databricks_run_id})


def _jobs_runs_submit(req_body_json):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/submit", method="POST", json=req_body_json)


def _get_databricks_run_cmd(dbfs_fuse_tar_uri, run_id, entry_point, parameters):
    """
    Generates MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks
    """
    # Strip ".gz" and ".tar" file extensions from base filename of the tarfile
    tar_hash = os.path.splitext(os.path.splitext(os.path.basename(dbfs_fuse_tar_uri))[0])[0]
    container_tar_path = os.path.abspath(os.path.join(DB_TARFILE_BASE,
                                                      os.path.basename(dbfs_fuse_tar_uri)))
    project_dir = os.path.join(DB_PROJECTS_BASE, tar_hash)
    mlflow_run_arr = list(map(shlex_quote, ["mlflow", "run", project_dir,
                                            "--entry-point", entry_point]))
    if run_id:
        mlflow_run_arr.extend(["--run-id", run_id])
    if parameters:
        for key, value in parameters.items():
            mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    mlflow_run_cmd = " ".join(mlflow_run_arr)
    shell_command = textwrap.dedent("""
    export PATH=$DB_HOME/conda/bin:$DB_HOME/python/bin:$PATH &&
    mlflow --version &&
    # Make local directories in the container into which to copy/extract the tarred project
    mkdir -p {tarfile_base} {projects_base} &&
    # Rsync from DBFS FUSE to avoid copying archive into local filesystem if it already exists
    rsync -a -v --ignore-existing {dbfs_fuse_tar_path} {tarfile_base} &&
    # Extract project into a temporary directory. We don't extract directly into the desired
    # directory as tar extraction isn't guaranteed to be atomic
    cd $(mktemp -d) &&
    tar -xzvf {container_tar_path} &&
    # Atomically move the extracted project into the desired directory
    mv -T {tarfile_archive_name} {work_dir} &&
    {mlflow_run}
    """.format(tarfile_base=DB_TARFILE_BASE, projects_base=DB_PROJECTS_BASE,
               dbfs_fuse_tar_path=dbfs_fuse_tar_uri, container_tar_path=container_tar_path,
               tarfile_archive_name=DB_TARFILE_ARCHIVE_NAME, work_dir=project_dir,
               mlflow_run=mlflow_run_cmd))
    return ["bash", "-c", shell_command]


def _check_databricks_auth_available():
    try:
        process.exec_cmd(["databricks", "--version"])
    except process.ShellCommandException:
        raise ExecutionException(
            "Could not find Databricks CLI on PATH. Please install and configure the Databricks "
            "CLI as described in https://github.com/databricks/databricks-cli")
    # Verify that we can get Databricks auth
    rest_utils.get_databricks_http_request_kwargs_or_fail()


def _upload_to_dbfs(src_path, dbfs_uri):
    """
    Uploads the file at `src_path` to the specified DBFS URI within the Databricks workspace
    corresponding to the default Databricks CLI profile.
    """
    eprint("=== Uploading project to DBFS path %s ===" % dbfs_uri)
    process.exec_cmd(cmd=["databricks", "fs", "cp", src_path, dbfs_uri])


def _dbfs_path_exists(dbfs_uri):
    """
    Returns True if the passed-in path exists in DBFS for the workspace corresponding to the
    default Databricks CLI profile.
    """
    dbfs_path = _parse_dbfs_uri_path(dbfs_uri)
    json_response_obj = rest_utils.databricks_api_request(
        endpoint="dbfs/get-status", method="GET", json={"path": dbfs_path})
    # If request fails with a RESOURCE_DOES_NOT_EXIST error, the file does not exist on DBFS
    error_code_field = "error_code"
    if error_code_field in json_response_obj:
        if json_response_obj[error_code_field] == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise ExecutionException("Got unexpected error response when checking whether file %s "
                                 "exists in DBFS: %s" % json_response_obj)
    return True


def _upload_project_to_dbfs(project_dir, experiment_id):
    """
    Tars a project directory into an archive in a temp dir and uploads it to DBFS, returning
    the HDFS-style URI of the tarball in DBFS (e.g. dbfs:/path/to/tar).

    :param project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                        a directory containing an MLproject file).
    """
    temp_tarfile_dir = tempfile.mkdtemp()
    temp_tar_filename = file_utils.build_path(temp_tarfile_dir, "project.tar.gz")
    try:
        file_utils.make_tarfile(temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME)
        with open(temp_tar_filename, "rb") as tarred_project:
            tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
        # TODO: Get subdirectory for experiment from the tracking server
        dbfs_uri = os.path.join("dbfs:/", DBFS_EXPERIMENT_DIR_BASE, str(experiment_id),
                                "projects-code", "%s.tar.gz" % tarfile_hash)
        if not _dbfs_path_exists(dbfs_uri):
            _upload_to_dbfs(temp_tar_filename, dbfs_uri)
            eprint("=== Finished uploading project to %s ===" % dbfs_uri)
        else:
            eprint("=== Project already exists in DBFS ===")
    finally:
        shutil.rmtree(temp_tarfile_dir)
    return dbfs_uri


def _get_run_result_state(databricks_run_id):
    """
    Returns the run result state (string) of the Databricks run with the passed-in ID, or None
    if the run is still active. See possible values at
    https://docs.databricks.com/api/latest/jobs.html#runresultstate.
    """
    res = _jobs_runs_get(databricks_run_id)
    return res["state"].get("result_state", None)


def _run_shell_command_job(project_uri, command, env_vars, cluster_spec):
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
    run_submit_res = _jobs_runs_submit(req_body_json)
    databricks_run_id = run_submit_res["run_id"]
    eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
           "page URL... ===" % databricks_run_id)
    run_info = _jobs_runs_get(databricks_run_id)
    jobs_page_url = run_info["run_page_url"]
    eprint("=== Check the run's status at %s ===" % jobs_page_url)
    return databricks_run_id


def _parse_dbfs_uri_path(dbfs_uri):
    """
    Parses and returns the absolute path within DBFS of the file with the specified URI. For
    example, given an input of "dbfs:/my/dbfs/path", this method will return "/my/dbfs/path"
    """
    return urllib.parse.urlparse(dbfs_uri).path


def _fetch_and_clean_project(uri, version=None, git_username=None, git_password=None):
    """
    Fetches the project at the passed-in URI & prepares it for upload to DBFS. Returns the path of
    the temporary directory into which the project was fetched.
    """
    work_dir = _fetch_project(
        uri=uri, force_tempdir=True, version=version, git_username=git_username,
        git_password=git_password)
    # Remove the mlruns directory from the fetched project to avoid cache-busting
    mlruns_dir = os.path.join(work_dir, "mlruns")
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
    return work_dir


def _before_run_validations(tracking_uri, cluster_spec):
    """Validations to perform before running a project on Databricks."""
    _check_databricks_auth_available()
    if cluster_spec is None:
        raise ExecutionException("Cluster spec must be provided when launching MLflow project runs "
                                 "on Databricks.")
    if tracking.is_local_uri(tracking_uri):
        raise ExecutionException(
            "When running on Databricks, the MLflow tracking URI must be set to a remote URI "
            "accessible to both the current client and code running on Databricks. Got local "
            "tracking URI %s." % tracking_uri)


def run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec,
                   git_username, git_password):
    """
    Runs the project at the specified URI on Databricks, returning a `SubmittedRun` that can be
    used to query the run's status or wait for the resulting Databricks Job run to terminate.
    """
    tracking_uri = tracking.get_tracking_uri()
    _before_run_validations(tracking_uri, cluster_spec)
    work_dir = _fetch_and_clean_project(
        uri=uri, version=version, git_username=git_username, git_password=git_password)
    project = _load_project(work_dir)
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    dbfs_project_uri = _upload_project_to_dbfs(work_dir, experiment_id)
    remote_run = tracking._create_run(
        experiment_id=experiment_id, source_name=_expand_uri(uri),
        source_version=tracking._get_git_commit(work_dir), entry_point_name=entry_point,
        source_type=SourceType.PROJECT)
    env_vars = {
         tracking._TRACKING_URI_ENV_VAR: tracking_uri,
         tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
    }
    run_id = remote_run.run_info.run_uuid
    eprint("=== Running entry point %s of project %s on Databricks. ===" % (entry_point, uri))
    # Launch run on Databricks
    with open(cluster_spec, 'r') as handle:
        try:
            cluster_spec = json.load(handle)
        except ValueError:
            eprint("Error when attempting to load and parse JSON cluster spec from file "
                   "%s. " % cluster_spec)
            raise
    fuse_dst_dir = os.path.join("/dbfs/", _parse_dbfs_uri_path(dbfs_project_uri).lstrip("/"))
    command = _get_databricks_run_cmd(fuse_dst_dir, run_id, entry_point, parameters)
    db_run_id = _run_shell_command_job(uri, command, env_vars, cluster_spec)
    return DatabricksSubmittedRun(db_run_id, run_id)


def _cancel_databricks(databricks_run_id):
    _jobs_runs_cancel(databricks_run_id)


def _monitor_databricks(databricks_run_id, sleep_interval=30):
    """
    Polls a Databricks Job run (with run ID `databricks_run_id`) for termination, checking the
    run's status every `sleep_interval` seconds.
    """
    result_state = _get_run_result_state(databricks_run_id)
    while result_state is None:
        time.sleep(sleep_interval)
        result_state = _get_run_result_state(databricks_run_id)
    return result_state == "SUCCESS"


class DatabricksSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Databricks Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.
    """
    def __init__(self, databricks_run_id, run_id):
        super(DatabricksSubmittedRun, self).__init__()
        self.databricks_run_id = databricks_run_id
        self.run_id = run_id

    def wait(self):
        return _monitor_databricks(self.databricks_run_id)

    def cancel(self):
        _cancel_databricks(self.databricks_run_id)
        self.wait()

    def _get_status(self):
        run_state = _get_run_result_state(self.databricks_run_id)
        if run_state is None:
            return RunStatus.RUNNING
        if run_state == "SUCCESS":
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())
