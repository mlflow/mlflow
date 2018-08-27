import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time

from six.moves import shlex_quote, urllib

from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.utils import rest_utils, file_utils
from mlflow.utils.exception import ExecutionException
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


class DatabricksJobRunner(object):
    """Helper class for running an MLflow project as a Databricks Job."""
    def __init__(self, profile):
        self.profile = profile

    def _jobs_runs_submit(self, json):
        return rest_utils.databricks_api_request(
            endpoint="jobs/runs/submit", method="POST", json=json, profile=self.profile)

    def _check_auth_available(self):
        """
        Verifies that information for making API requests to Databricks is available to MLflow,
        raising an exception if not.
        """
        rest_utils.get_databricks_http_request_kwargs_or_fail(self.profile)

    def _upload_to_dbfs(self, src_path, dbfs_fuse_uri):
        """
        Uploads the file at `src_path` to the specified DBFS URI within the Databricks workspace
        corresponding to the default Databricks CLI profile.
        """
        eprint("=== Uploading project to DBFS path %s ===" % dbfs_fuse_uri)
        http_endpoint = dbfs_fuse_uri
        http_request_kwargs = rest_utils.get_databricks_http_request_kwargs_or_fail(self.profile)
        with open(src_path, 'rb') as f:
            rest_utils.http_request(
                endpoint=http_endpoint, method='POST', data=f, **http_request_kwargs)

    def _dbfs_path_exists(self, dbfs_uri):
        """
        Returns True if the passed-in path exists in DBFS for the workspace corresponding to the
        default Databricks CLI profile.
        """
        dbfs_path = _parse_dbfs_uri_path(dbfs_uri)
        json_response_obj = rest_utils.databricks_api_request(
            endpoint="dbfs/get-status", method="GET", json={"path": dbfs_path},
            profile=self.profile)
        # If request fails with a RESOURCE_DOES_NOT_EXIST error, the file does not exist on DBFS
        error_code_field = "error_code"
        if error_code_field in json_response_obj:
            if json_response_obj[error_code_field] == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise ExecutionException("Got unexpected error response when checking whether file %s "
                                     "exists in DBFS: %s" % json_response_obj)
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
            dbfs_fuse_uri = os.path.join("/dbfs", DBFS_EXPERIMENT_DIR_BASE, str(experiment_id),
                                         "projects-code", "%s.tar.gz" % tarfile_hash)
            if not self._dbfs_path_exists(dbfs_fuse_uri):
                self._upload_to_dbfs(temp_tar_filename, dbfs_fuse_uri)
                eprint("=== Finished uploading project to %s ===" % dbfs_fuse_uri)
            else:
                eprint("=== Project already exists in DBFS ===")
        finally:
            shutil.rmtree(temp_tarfile_dir)
        return dbfs_fuse_uri

    def _run_shell_command_job(self, project_uri, command, env_vars, cluster_spec):
        """
        Runs the specified shell command on a Databricks cluster.
        :param project_uri: URI of the project from which our shell command originates
        :param command: Shell command to run
        :param env_vars: Environment variables to set in the process running `command`
        :param cluster_spec: Dictionary describing the cluster, expected to contain the fields for a
                             NewCluster (see https://docs.databricks.com/api/latest/
                             jobs.html#jobsclusterspecnewcluster)
        :return: The ID of the Databricks Job Run. Can be used to query the run's status via the
                 Databricks Runs Get API
                 (https://docs.databricks.com/api/latest/jobs.html#runs-get).
        """
        # Make jobs API request to launch run.
        req_body_json = {
            'run_name': 'MLflow Run for %s' % project_uri,
            'new_cluster': cluster_spec,
            'shell_command_task': {
                'command': command,
                "env_vars": env_vars
            },
            "libraries": [{"pypi": {"package": "'mlflow<=%s'" % VERSION}}]
        }
        run_submit_res = self._jobs_runs_submit(req_body_json)
        databricks_run_id = run_submit_res["run_id"]
        return databricks_run_id

    def _before_run_validations(self, tracking_uri, cluster_spec):
        """Validations to perform before running a project on Databricks."""
        self._check_auth_available()
        if cluster_spec is None:
            raise ExecutionException("Cluster spec must be provided when launching MLflow project "
                                     "runs on Databricks.")
        if tracking.utils._is_local_uri(tracking_uri):
            raise ExecutionException(
                "When running on Databricks, the MLflow tracking URI must be set to a remote URI "
                "accessible to both the current client and code running on Databricks. Got local "
                "tracking URI %s." % tracking_uri)

    def run_databricks(self, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec,
                       run_id):
        tracking_uri = tracking.get_tracking_uri()
        self._before_run_validations(tracking_uri, cluster_spec)
        dbfs_fuse_uri = self._upload_project_to_dbfs(work_dir, experiment_id)
        env_vars = {
            tracking._TRACKING_URI_ENV_VAR: tracking_uri,
            tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
        }
        eprint("=== Running entry point %s of project %s on Databricks ===" % (entry_point, uri))
        # Launch run on Databricks
        with open(cluster_spec, 'r') as handle:
            try:
                cluster_spec = json.load(handle)
            except ValueError:
                eprint("Error when attempting to load and parse JSON cluster spec from file "
                       "%s. " % cluster_spec)
                raise
        command = _get_databricks_run_cmd(dbfs_fuse_uri, run_id, entry_point, parameters)
        return self._run_shell_command_job(uri, command, env_vars, cluster_spec)


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
    tar --no-same-owner -xzvf {container_tar_path} &&
    # Atomically move the extracted project into the desired directory
    mv -T {tarfile_archive_name} {work_dir} &&
    {mlflow_run}
    """.format(tarfile_base=DB_TARFILE_BASE, projects_base=DB_PROJECTS_BASE,
               dbfs_fuse_tar_path=dbfs_fuse_tar_uri, container_tar_path=container_tar_path,
               tarfile_archive_name=DB_TARFILE_ARCHIVE_NAME, work_dir=project_dir,
               mlflow_run=mlflow_run_cmd))
    return ["bash", "-c", shell_command]


def _parse_dbfs_uri_path(dbfs_uri):
    """
    Parses and returns the absolute path within DBFS of the file with the specified URI. For
    example, given an input of "dbfs:/my/dbfs/path", this method will return "/my/dbfs/path"
    """
    return urllib.parse.urlparse(dbfs_uri).path


def run_databricks(remote_run, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec):
    """
    Runs the project at the specified URI on Databricks, returning a `SubmittedRun` that can be
    used to query the run's status or wait for the resulting Databricks Job run to terminate.
    """
    tracking_uri = tracking.get_tracking_uri()
    profile = tracking.utils.get_db_profile_from_uri(tracking_uri)
    run_id = remote_run.info.run_uuid
    db_job_runner = DatabricksJobRunner(profile=profile)
    db_run_id = db_job_runner.run_databricks(
        uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, run_id)
    submitted_run = DatabricksSubmittedRun(db_run_id, run_id, profile)
    submitted_run._print_description()
    return submitted_run


class DatabricksSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Databricks Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.
    """
    def __init__(self, databricks_run_id, run_id, profile):
        super(DatabricksSubmittedRun, self).__init__()
        self._databricks_run_id = databricks_run_id
        self._run_id = run_id
        self._profile = profile

    def _print_description(self):
        eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
               "page URL... ===" % self._databricks_run_id)
        run_info = self._jobs_runs_get(self._databricks_run_id)
        jobs_page_url = run_info["run_page_url"]
        eprint("=== Check the run's status at %s ===" % jobs_page_url)

    @property
    def run_id(self):
        return self._run_id

    def wait(self, sleep_interval=30):
        result_state = self._get_run_result_state(self._databricks_run_id)
        while result_state is None:
            time.sleep(sleep_interval)
            result_state = self._get_run_result_state(self._databricks_run_id)
        return result_state == "SUCCESS"

    def cancel(self):
        self._jobs_runs_cancel(self._databricks_run_id)
        self.wait()

    def _get_status(self):
        run_state = self._get_run_result_state(self._databricks_run_id)
        if run_state is None:
            return RunStatus.RUNNING
        if run_state == "SUCCESS":
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())

    def _get_run_result_state(self, databricks_run_id):
        """
        Returns the run result state (string) of the Databricks run with the passed-in ID, or None
        if the run is still active. See possible values at
        https://docs.databricks.com/api/latest/jobs.html#runresultstate.
        """
        res = self._jobs_runs_get(databricks_run_id)
        return res["state"].get("result_state", None)

    def _jobs_runs_cancel(self, databricks_run_id):
        return rest_utils.databricks_api_request(
            endpoint="jobs/runs/cancel", method="POST", json={"run_id": databricks_run_id},
            profile=self._profile)

    def _jobs_runs_get(self, databricks_run_id):
        return rest_utils.databricks_api_request(
            endpoint="jobs/runs/get", method="GET", json={"run_id": databricks_run_id},
            profile=self._profile)
