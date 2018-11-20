import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time
import logging

from six.moves import shlex_quote

from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.utils import rest_utils, file_utils, databricks_utils
from mlflow.exceptions import ExecutionException
from mlflow import tracking
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_RUN_URL, MLFLOW_DATABRICKS_SHELL_JOB_ID, \
    MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, MLFLOW_DATABRICKS_WEBAPP_URL
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


_logger = logging.getLogger(__name__)


def before_run_validations(tracking_uri, cluster_spec):
    """Validations to perform before running a project on Databricks."""
    if cluster_spec is None:
        raise ExecutionException("Cluster spec must be provided when launching MLflow project "
                                 "runs on Databricks.")
    if tracking.utils._is_local_uri(tracking_uri):
        raise ExecutionException(
            "When running on Databricks, the MLflow tracking URI must be of the form "
            "'databricks' or 'databricks://profile', or a remote HTTP URI accessible to both the "
            "current client and code running on Databricks. Got local tracking URI %s. "
            "Please specify a valid tracking URI via mlflow.set_tracking_uri or by setting the "
            "MLFLOW_TRACKING_URI environment variable." % tracking_uri)


class DatabricksJobRunner(object):
    """
    Helper class for running an MLflow project as a Databricks Job.
    :param databricks_profile: Optional Databricks CLI profile to use to fetch hostname &
           authentication information when making Databricks API requests.
    """
    def __init__(self, databricks_profile):
        self.databricks_profile = databricks_profile

    def _databricks_api_request(self, endpoint, method, **kwargs):
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile)
        return rest_utils.http_request_safe(
            host_creds=host_creds, endpoint=endpoint, method=method, **kwargs)

    def _jobs_runs_submit(self, req_body):
        response = self._databricks_api_request(
            endpoint="/api/2.0/jobs/runs/submit", method="POST", json=req_body)
        return json.loads(response.text)

    def _upload_to_dbfs(self, src_path, dbfs_fuse_uri):
        """
        Upload the file at `src_path` to the specified DBFS URI within the Databricks workspace
        corresponding to the default Databricks CLI profile.
        """
        _logger.info("=== Uploading project to DBFS path %s ===", dbfs_fuse_uri)
        http_endpoint = dbfs_fuse_uri
        with open(src_path, 'rb') as f:
            self._databricks_api_request(endpoint=http_endpoint, method='POST', data=f)

    def _dbfs_path_exists(self, dbfs_path):
        """
        Return True if the passed-in path exists in DBFS for the workspace corresponding to the
        default Databricks CLI profile. The path is expected to be a relative path to the DBFS root
        directory, e.g. 'path/to/file'.
        """
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile)
        response = rest_utils.http_request(
            host_creds=host_creds, endpoint="/api/2.0/dbfs/get-status", method="GET",
            json={"path": "/%s" % dbfs_path})
        try:
            json_response_obj = json.loads(response.text)
        except ValueError:
            raise MlflowException(
                "API request to check existence of file at DBFS path %s failed with status code "
                "%s. Response body: %s" % (dbfs_path, response.status_code, response.text))
        # If request fails with a RESOURCE_DOES_NOT_EXIST error, the file does not exist on DBFS
        error_code_field = "error_code"
        if error_code_field in json_response_obj:
            if json_response_obj[error_code_field] == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise ExecutionException("Got unexpected error response when checking whether file %s "
                                     "exists in DBFS: %s" % (dbfs_path, json_response_obj))
        return True

    def _upload_project_to_dbfs(self, project_dir, experiment_id):
        """
        Tars a project directory into an archive in a temp dir and uploads it to DBFS, returning
        the HDFS-style URI of the tarball in DBFS (e.g. dbfs:/path/to/tar).

        :param project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                            a directory containing an MLproject file).
        """
        temp_tarfile_dir = tempfile.mkdtemp()
        temp_tar_filename = file_utils.build_path(temp_tarfile_dir, "project.tar.gz")

        def custom_filter(x):
            return None if os.path.basename(x.name) == "mlruns" else x

        try:
            file_utils.make_tarfile(temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME,
                                    custom_filter=custom_filter)
            with open(temp_tar_filename, "rb") as tarred_project:
                tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
            # TODO: Get subdirectory for experiment from the tracking server
            dbfs_path = os.path.join(DBFS_EXPERIMENT_DIR_BASE, str(experiment_id),
                                     "projects-code", "%s.tar.gz" % tarfile_hash)
            dbfs_fuse_uri = os.path.join("/dbfs", dbfs_path)
            if not self._dbfs_path_exists(dbfs_path):
                self._upload_to_dbfs(temp_tar_filename, dbfs_fuse_uri)
                _logger.info("=== Finished uploading project to %s ===", dbfs_fuse_uri)
            else:
                _logger.info("=== Project already exists in DBFS ===")
        finally:
            shutil.rmtree(temp_tarfile_dir)
        return dbfs_fuse_uri

    def _run_shell_command_job(self, project_uri, command, env_vars, cluster_spec):
        """
        Run the specified shell command on a Databricks cluster.

        :param project_uri: URI of the project from which the shell command originates.
        :param command: Shell command to run.
        :param env_vars: Environment variables to set in the process running ``command``.
        :param cluster_spec: Dictionary containing a `Databricks cluster specification
                             <https://docs.databricks.com/api/latest/jobs.html#clusterspec>`_
                             to use when launching a run.
        :return: ID of the Databricks job run. Can be used to query the run's status via the
                 Databricks
                 `Runs Get <https://docs.databricks.com/api/latest/jobs.html#runs-get>`_ API.
        """
        # Make jobs API request to launch run.
        req_body_json = {
            'run_name': 'MLflow Run for %s' % project_uri,
            'new_cluster': cluster_spec,
            'shell_command_task': {
                'command': command,
                "env_vars": env_vars
            },
            # NB: We use <= on the version specifier to allow running projects on pre-release
            # versions, where we will select the most up-to-date mlflow version available.
            # Also note, that we escape this so '<' is not treated as a shell pipe.
            "libraries": [{"pypi": {"package": "'mlflow<=%s'" % VERSION}}],
        }
        run_submit_res = self._jobs_runs_submit(req_body_json)
        databricks_run_id = run_submit_res["run_id"]
        return databricks_run_id

    def run_databricks(self, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec,
                       run_id):
        tracking_uri = _get_tracking_uri_for_run()
        dbfs_fuse_uri = self._upload_project_to_dbfs(work_dir, experiment_id)
        env_vars = {
            tracking._TRACKING_URI_ENV_VAR: tracking_uri,
            tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
        }
        _logger.info("=== Running entry point %s of project %s on Databricks ===", entry_point, uri)
        # Launch run on Databricks
        command = _get_databricks_run_cmd(dbfs_fuse_uri, run_id, entry_point, parameters)
        return self._run_shell_command_job(uri, command, env_vars, cluster_spec)

    def _get_status(self, databricks_run_id):
        run_state = self.get_run_result_state(databricks_run_id)
        if run_state is None:
            return RunStatus.RUNNING
        if run_state == "SUCCESS":
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self, databricks_run_id):
        return RunStatus.to_string(self._get_status(databricks_run_id))

    def get_run_result_state(self, databricks_run_id):
        """
        Get the run result state (string) of a Databricks job run.

        :param databricks_run_id: Integer Databricks job run ID.
        :returns `RunResultState
        <https://docs.databricks.com/api/latest/jobs.html#runresultstate>`_ or None if
        the run is still active.
        """
        res = self.jobs_runs_get(databricks_run_id)
        return res["state"].get("result_state", None)

    def jobs_runs_cancel(self, databricks_run_id):
        response = self._databricks_api_request(
            endpoint="/api/2.0/jobs/runs/cancel", method="POST", json={"run_id": databricks_run_id})
        return json.loads(response.text)

    def jobs_runs_get(self, databricks_run_id):
        response = self._databricks_api_request(
            endpoint="/api/2.0/jobs/runs/get", method="GET", json={"run_id": databricks_run_id})
        return json.loads(response.text)


def _get_tracking_uri_for_run():
    uri = tracking.get_tracking_uri()
    if uri.startswith("databricks"):
        return "databricks"
    return uri


def _get_databricks_run_cmd(dbfs_fuse_tar_uri, run_id, entry_point, parameters):
    """
    Generate MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks.
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
    tar --no-same-owner -xzvf {container_tar_path} &&
    # Atomically move the extracted project into the desired directory
    mv -T {tarfile_archive_name} {work_dir} &&
    {mlflow_run}
    """.format(tarfile_base=DB_TARFILE_BASE, projects_base=DB_PROJECTS_BASE,
               dbfs_fuse_tar_path=dbfs_fuse_tar_uri, container_tar_path=container_tar_path,
               tarfile_archive_name=DB_TARFILE_ARCHIVE_NAME, work_dir=project_dir,
               mlflow_run=mlflow_run_cmd))
    return ["bash", "-c", shell_command]


def run_databricks(remote_run, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec):
    """
    Run the project at the specified URI on Databricks, returning a ``SubmittedRun`` that can be
    used to query the run's status or wait for the resulting Databricks Job run to terminate.
    """
    profile = tracking.utils.get_db_profile_from_uri(tracking.get_tracking_uri())
    run_id = remote_run.info.run_uuid
    db_job_runner = DatabricksJobRunner(databricks_profile=profile)
    db_run_id = db_job_runner.run_databricks(
        uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, run_id)
    submitted_run = DatabricksSubmittedRun(db_run_id, run_id, db_job_runner)
    submitted_run._print_description_and_log_tags()
    return submitted_run


class DatabricksSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Databricks Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.
    :param databricks_run_id: Run ID of the launched Databricks Job.
    :param mlflow_run_id: ID of the MLflow project run.
    :param databricks_job_runner: Instance of ``DatabricksJobRunner`` used to make Databricks API
                                  requests.
    """
    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 30

    def __init__(self, databricks_run_id, mlflow_run_id, databricks_job_runner):
        super(DatabricksSubmittedRun, self).__init__()
        self._databricks_run_id = databricks_run_id
        self._mlflow_run_id = mlflow_run_id
        self._job_runner = databricks_job_runner

    def _print_description_and_log_tags(self):
        _logger.info(
            "=== Launched MLflow run as Databricks job run with ID %s."
            " Getting run status page URL... ===",
            self._databricks_run_id)
        run_info = self._job_runner.jobs_runs_get(self._databricks_run_id)
        jobs_page_url = run_info["run_page_url"]
        _logger.info("=== Check the run's status at %s ===", jobs_page_url)
        host_creds = databricks_utils.get_databricks_host_creds(self._job_runner.databricks_profile)
        tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                        MLFLOW_DATABRICKS_RUN_URL, jobs_page_url)
        tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                        MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, self._databricks_run_id)
        tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                        MLFLOW_DATABRICKS_WEBAPP_URL, host_creds.host)
        job_id = run_info.get('job_id')
        # In some releases of Databricks we do not return the job ID. We start including it in DB
        # releases 2.80 and above.
        if job_id is not None:
            tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                            MLFLOW_DATABRICKS_SHELL_JOB_ID, job_id)

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
        while result_state is None:
            time.sleep(self.POLL_STATUS_INTERVAL)
            result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
        return result_state == "SUCCESS"

    def cancel(self):
        self._job_runner.jobs_runs_cancel(self._databricks_run_id)
        self.wait()

    def get_status(self):
        return self._job_runner.get_status(self._databricks_run_id)
