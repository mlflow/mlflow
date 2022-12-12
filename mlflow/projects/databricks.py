import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time
import logging
import posixpath
import re

from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import rest_utils, file_utils, databricks_utils
from mlflow.exceptions import ExecutionException
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_RUN_URL,
    MLFLOW_DATABRICKS_SHELL_JOB_ID,
    MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID,
    MLFLOW_DATABRICKS_WEBAPP_URL,
)
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.utils.string_utils import quote
from mlflow.version import is_release_version, VERSION

# Base directory within driver container for storing files related to MLflow
DB_CONTAINER_BASE = "/databricks/mlflow"
# Base directory within driver container for storing project archives
DB_TARFILE_BASE = posixpath.join(DB_CONTAINER_BASE, "project-tars")
# Base directory directory within driver container for storing extracted project directories
DB_PROJECTS_BASE = posixpath.join(DB_CONTAINER_BASE, "projects")
# Name to use for project directory when archiving it for upload to DBFS; the TAR will contain
# a single directory with this name
DB_TARFILE_ARCHIVE_NAME = "mlflow-project"
# Base directory within DBFS for storing code for project runs for experiments
DBFS_EXPERIMENT_DIR_BASE = "mlflow-experiments"


_logger = logging.getLogger(__name__)

_MLFLOW_GIT_URI_REGEX = re.compile(r"^git\+https://github.com/[\w-]+/mlflow")


def _is_mlflow_git_uri(s):
    return bool(_MLFLOW_GIT_URI_REGEX.match(s))


def _contains_mlflow_git_uri(libraries):
    for lib in libraries:
        package = lib.get("pypi", {}).get("package")
        if package and _is_mlflow_git_uri(package):
            return True
    return False


def before_run_validations(tracking_uri, backend_config):
    """Validations to perform before running a project on Databricks."""
    if backend_config is None:
        raise ExecutionException(
            "Backend spec must be provided when launching MLflow project runs on Databricks."
        )
    elif "existing_cluster_id" in backend_config:
        raise MlflowException(
            message=(
                "MLflow Project runs on Databricks must provide a *new cluster* specification."
                " Project execution against existing clusters is not currently supported. For more"
                " information, see https://mlflow.org/docs/latest/projects.html"
                "#run-an-mlflow-project-on-databricks"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not is_databricks_uri(tracking_uri) and not is_http_uri(tracking_uri):
        raise ExecutionException(
            "When running on Databricks, the MLflow tracking URI must be of the form "
            "'databricks' or 'databricks://profile', or a remote HTTP URI accessible to both the "
            "current client and code running on Databricks. Got local tracking URI %s. "
            "Please specify a valid tracking URI via mlflow.set_tracking_uri or by setting the "
            "MLFLOW_TRACKING_URI environment variable." % tracking_uri
        )


class DatabricksJobRunner:
    """
    Helper class for running an MLflow project as a Databricks Job.
    :param databricks_profile: Optional Databricks CLI profile to use to fetch hostname &
           authentication information when making Databricks API requests.
    """

    def __init__(self, databricks_profile_uri):
        self.databricks_profile_uri = databricks_profile_uri

    def _databricks_api_request(self, endpoint, method, **kwargs):
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile_uri)
        return rest_utils.http_request_safe(
            host_creds=host_creds, endpoint=endpoint, method=method, **kwargs
        )

    def _jobs_runs_submit(self, req_body):
        response = self._databricks_api_request(
            endpoint="/api/2.0/jobs/runs/submit", method="POST", json=req_body
        )
        return json.loads(response.text)

    def _upload_to_dbfs(self, src_path, dbfs_fuse_uri):
        """
        Upload the file at `src_path` to the specified DBFS URI within the Databricks workspace
        corresponding to the default Databricks CLI profile.
        """
        _logger.info("=== Uploading project to DBFS path %s ===", dbfs_fuse_uri)
        http_endpoint = dbfs_fuse_uri
        with open(src_path, "rb") as f:
            try:
                self._databricks_api_request(endpoint=http_endpoint, method="POST", data=f)
            except MlflowException as e:
                if "Error 409" in e.message and "File already exists" in e.message:
                    _logger.info("=== Did not overwrite existing DBFS path %s ===", dbfs_fuse_uri)
                else:
                    raise e

    def _dbfs_path_exists(self, dbfs_path):
        """
        Return True if the passed-in path exists in DBFS for the workspace corresponding to the
        default Databricks CLI profile. The path is expected to be a relative path to the DBFS root
        directory, e.g. 'path/to/file'.
        """
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile_uri)
        response = rest_utils.http_request(
            host_creds=host_creds,
            endpoint="/api/2.0/dbfs/get-status",
            method="GET",
            json={"path": "/%s" % dbfs_path},
        )
        try:
            json_response_obj = json.loads(response.text)
        except Exception:
            raise MlflowException(
                "API request to check existence of file at DBFS path %s failed with status code "
                "%s. Response body: %s" % (dbfs_path, response.status_code, response.text)
            )
        # If request fails with a RESOURCE_DOES_NOT_EXIST error, the file does not exist on DBFS
        error_code_field = "error_code"
        if error_code_field in json_response_obj:
            if json_response_obj[error_code_field] == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise ExecutionException(
                "Got unexpected error response when checking whether file %s "
                "exists in DBFS: %s" % (dbfs_path, json_response_obj)
            )
        return True

    def _upload_project_to_dbfs(self, project_dir, experiment_id):
        """
        Tars a project directory into an archive in a temp dir and uploads it to DBFS, returning
        the HDFS-style URI of the tarball in DBFS (e.g. dbfs:/path/to/tar).

        :param project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                            a directory containing an MLproject file).
        """
        temp_tarfile_dir = tempfile.mkdtemp()
        temp_tar_filename = os.path.join(temp_tarfile_dir, "project.tar.gz")

        def custom_filter(x):
            return None if os.path.basename(x.name) == "mlruns" else x

        try:
            directory_size = file_utils._get_local_project_dir_size(project_dir)
            _logger.info(
                f"=== Creating tarball from {project_dir} in temp directory {temp_tarfile_dir} ==="
            )
            _logger.info(f"=== Total file size to compress: {directory_size} KB ===")
            file_utils.make_tarfile(
                temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME, custom_filter=custom_filter
            )
            with open(temp_tar_filename, "rb") as tarred_project:
                tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
            # TODO: Get subdirectory for experiment from the tracking server
            dbfs_path = posixpath.join(
                DBFS_EXPERIMENT_DIR_BASE,
                str(experiment_id),
                "projects-code",
                "%s.tar.gz" % tarfile_hash,
            )
            tar_size = file_utils._get_local_file_size(temp_tar_filename)
            dbfs_fuse_uri = posixpath.join("/dbfs", dbfs_path)
            if not self._dbfs_path_exists(dbfs_path):
                _logger.info(
                    f"=== Uploading project tarball (size: {tar_size} KB) to {dbfs_fuse_uri} ==="
                )
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
                             <https://docs.databricks.com/dev-tools/api/latest/jobs.html#clusterspec>`_
                             or a `Databricks new cluster specification
                             <https://docs.databricks.com/dev-tools/api/latest/jobs.html#jobsclusterspecnewcluster>`_
                             to use when launching a run. If you specify libraries, this function
                             will add MLflow to the library list. This function does not support
                             installation of conda environment libraries on the workers.
        :return: ID of the Databricks job run. Can be used to query the run's status via the
                 Databricks
                 `Runs Get <https://docs.databricks.com/api/latest/jobs.html#runs-get>`_ API.
        """
        if is_release_version():
            mlflow_lib = {"pypi": {"package": "mlflow==%s" % VERSION}}
        else:
            # When running a non-release version as the client the same version will not be
            # available within Databricks.
            _logger.warning(
                "Your client is running a non-release version of MLFlow. "
                "This version is not available on the databricks runtime. "
                "MLFlow will fallback the MLFlow version provided by the runtime. "
                "This might lead to unforeseen issues. "
            )
            mlflow_lib = {"pypi": {"package": "'mlflow<=%s'" % VERSION}}

        # Check syntax of JSON - if it contains libraries and new_cluster, pull those out
        if "new_cluster" in cluster_spec:
            # Libraries are optional, so we don't require that this be specified
            cluster_spec_libraries = cluster_spec.get("libraries", [])
            libraries = (
                # This is for development purposes only. If the cluster spec already includes
                # an MLflow Git URI, then we don't append `mlflow_lib` to avoid having
                # two different pip requirements for mlflow.
                cluster_spec_libraries
                if _contains_mlflow_git_uri(cluster_spec_libraries)
                else cluster_spec_libraries + [mlflow_lib]
            )
            cluster_spec = cluster_spec["new_cluster"]
        else:
            libraries = [mlflow_lib]

        # Make jobs API request to launch run.
        req_body_json = {
            "run_name": "MLflow Run for %s" % project_uri,
            "new_cluster": cluster_spec,
            "shell_command_task": {"command": command, "env_vars": env_vars},
            "libraries": libraries,
        }
        _logger.info("=== Submitting a run to execute the MLflow project... ===")
        run_submit_res = self._jobs_runs_submit(req_body_json)
        databricks_run_id = run_submit_res["run_id"]
        return databricks_run_id

    def run_databricks(
        self,
        uri,
        entry_point,
        work_dir,
        parameters,
        experiment_id,
        cluster_spec,
        run_id,
        env_manager,
    ):
        tracking_uri = _get_tracking_uri_for_run()
        dbfs_fuse_uri = self._upload_project_to_dbfs(work_dir, experiment_id)
        env_vars = {
            tracking._TRACKING_URI_ENV_VAR: tracking_uri,
            tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
        }
        _logger.info("=== Running entry point %s of project %s on Databricks ===", entry_point, uri)
        # Launch run on Databricks
        command = _get_databricks_run_cmd(
            dbfs_fuse_uri, run_id, entry_point, parameters, env_manager
        )
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
            endpoint="/api/2.0/jobs/runs/cancel", method="POST", json={"run_id": databricks_run_id}
        )
        return json.loads(response.text)

    def jobs_runs_get(self, databricks_run_id):
        response = self._databricks_api_request(
            endpoint="/api/2.0/jobs/runs/get", method="GET", params={"run_id": databricks_run_id}
        )
        return json.loads(response.text)


def _get_tracking_uri_for_run():
    uri = tracking.get_tracking_uri()
    if uri.startswith("databricks"):
        return "databricks"
    return uri


def _get_cluster_mlflow_run_cmd(project_dir, run_id, entry_point, parameters, env_manager):
    cmd = [
        "mlflow",
        "run",
        project_dir,
        "--entry-point",
        entry_point,
    ]
    if env_manager:
        cmd += ["--env-manager", env_manager]
    mlflow_run_arr = list(map(quote, cmd))
    if run_id:
        mlflow_run_arr.extend(["-c", json.dumps({MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG: run_id})])
    if parameters:
        for key, value in parameters.items():
            mlflow_run_arr.extend(["-P", f"{key}={value}"])
    return mlflow_run_arr


def _get_databricks_run_cmd(dbfs_fuse_tar_uri, run_id, entry_point, parameters, env_manager):
    """
    Generate MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks.
    """
    # Strip ".gz" and ".tar" file extensions from base filename of the tarfile
    tar_hash = posixpath.splitext(posixpath.splitext(posixpath.basename(dbfs_fuse_tar_uri))[0])[0]
    container_tar_path = posixpath.abspath(
        posixpath.join(DB_TARFILE_BASE, posixpath.basename(dbfs_fuse_tar_uri))
    )
    project_dir = posixpath.join(DB_PROJECTS_BASE, tar_hash)

    mlflow_run_arr = _get_cluster_mlflow_run_cmd(
        project_dir,
        run_id,
        entry_point,
        parameters,
        env_manager,
    )
    mlflow_run_cmd = " ".join([quote(elem) for elem in mlflow_run_arr])
    shell_command = textwrap.dedent(
        f"""
    export PATH=$PATH:$DB_HOME/python/bin &&
    mlflow --version &&
    # Make local directories in the container into which to copy/extract the tarred project
    mkdir -p {DB_TARFILE_BASE} {DB_PROJECTS_BASE} &&
    # Rsync from DBFS FUSE to avoid copying archive into local filesystem if it already exists
    rsync -a -v --ignore-existing {dbfs_fuse_tar_uri} {DB_TARFILE_BASE} &&
    # Extract project into a temporary directory. We don't extract directly into the desired
    # directory as tar extraction isn't guaranteed to be atomic
    cd $(mktemp -d) &&
    tar --no-same-owner -xzvf {container_tar_path} &&
    # Atomically move the extracted project into the desired directory
    mv -T {DB_TARFILE_ARCHIVE_NAME} {project_dir} &&
    {mlflow_run_cmd}
    """
    )
    return ["bash", "-c", shell_command]


def run_databricks(
    remote_run, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, env_manager
):
    """
    Run the project at the specified URI on Databricks, returning a ``SubmittedRun`` that can be
    used to query the run's status or wait for the resulting Databricks Job run to terminate.
    """
    run_id = remote_run.info.run_id
    db_job_runner = DatabricksJobRunner(databricks_profile_uri=tracking.get_tracking_uri())
    db_run_id = db_job_runner.run_databricks(
        uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, run_id, env_manager
    )
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
        super().__init__()
        self._databricks_run_id = databricks_run_id
        self._mlflow_run_id = mlflow_run_id
        self._job_runner = databricks_job_runner

    def _print_description_and_log_tags(self):
        _logger.info(
            "=== Launched MLflow run as Databricks job run with ID %s."
            " Getting run status page URL... ===",
            self._databricks_run_id,
        )
        run_info = self._job_runner.jobs_runs_get(self._databricks_run_id)
        jobs_page_url = run_info["run_page_url"]
        _logger.info("=== Check the run's status at %s ===", jobs_page_url)
        host_creds = databricks_utils.get_databricks_host_creds(
            self._job_runner.databricks_profile_uri
        )
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_DATABRICKS_RUN_URL, jobs_page_url
        )
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, self._databricks_run_id
        )
        tracking.MlflowClient().set_tag(
            self._mlflow_run_id, MLFLOW_DATABRICKS_WEBAPP_URL, host_creds.host
        )
        job_id = run_info.get("job_id")
        # In some releases of Databricks we do not return the job ID. We start including it in DB
        # releases 2.80 and above.
        if job_id is not None:
            tracking.MlflowClient().set_tag(
                self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_ID, job_id
            )

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
