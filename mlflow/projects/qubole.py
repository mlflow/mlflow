import boto3
import botocore
import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import time

from six.moves import shlex_quote

from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.utils import rest_utils, file_utils
from mlflow.exceptions import ExecutionException
from mlflow.utils.logging_utils import eprint
from mlflow import tracking
from mlflow.utils.mlflow_tags import MLFLOW_QUBOLE_COMMAND_URL, MLFLOW_QUBOLE_COMMAND_ID
from mlflow.version import VERSION

from qds_sdk.commands import ShellCommand
from qds_sdk.qubole import Qubole

QUBOLE_TARFILE_ARCHIVE_NAME = "mlflow-project"
QUBOLE_CONDA_HOME = "/usr/lib/a-4.2.0-py-3.5.3/"


class S3Utils(object):
    def __init__(self, conf):
        self.bucket = conf["s3_experiment_bucket"]
        self.base_path = conf["s3_experiment_base_path"]
    
    def _get_bucket(self):
        return boto3.session.Session()\
                    .resource('s3').Bucket(self.bucket)

    def _path_exists(self, path):
        """
        Returns True if the passed-in path exists in s3.
        """
        try:
            self._get_bucket().load(path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return True # The object does not exist.
            else:
                raise e # Something else has gone wrong.
        else:
            return False # The object exists.

    def _upload(self, src_path, path):
        """
        Uploads the file at `src_path` to the specified S3 path.
        """
        eprint("=== Uploading to S3 path %s ===" % path)
        self._get_bucket().upload_file(src_path, path)
    
    def upload_project(self, project_dir, experiment_id):
        """
        Tars a project directory into an archive in a temp dir and uploads it to S3, returning
        the URI of the tarball in S3 (e.g. s3:/path/to/tar).
        :param project_dir: Path to a directory containing an MLflow project to upload to S3 (e.g.
                            a directory containing an MLproject file).
        """
        temp_tarfile_dir = tempfile.mkdtemp()
        temp_tar_filename = file_utils.build_path(temp_tarfile_dir, "project.tar.gz")
        try:
            file_utils.make_tarfile(temp_tar_filename, project_dir, QUBOLE_TARFILE_ARCHIVE_NAME)
            with open(temp_tar_filename, "rb") as tarred_project:
                tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
            # TODO: Get subdirectory for experiment from the tracking server
            s3_path = os.path.join(self.base_path, str(experiment_id),
                                "projects-code", "%s.tar.gz" % tarfile_hash)
            if not self._path_exists(s3_path):
                self._upload(temp_tar_filename, s3_path)
                eprint("=== Finished uploading project to %s ===" % s3_path)
            else:
                eprint("=== Project already exists in S3 ===")
        finally:
            shutil.rmtree(temp_tarfile_dir)
        
        full_path = "s3://{}/{}".format(self.bucket, s3_path)

        return full_path
    
    def upload_script(self, script, experiment_id):
        """
        Stores the scrip in a temp file and uploads it to S3, returning
        the URI of the script in S3 (e.g. s3:/path/to/tar).
        :param script: String containing the commands to be run on QDS.
        """
        temp_dir = tempfile.mkdtemp()
        temp_filename = file_utils.build_path(temp_dir, "script.sh")
        try:
            file_utils.write_to(temp_filename, script)
            with open(temp_filename, "r") as f:
                script_hash = hashlib.sha256(f.read().encode('utf-8')).hexdigest()
            # TODO: Get subdirectory for experiment from the tracking server
            s3_path = os.path.join(self.base_path, str(experiment_id),
                                "script", "%s.sh" % script_hash)
            if not self._path_exists(s3_path):
                self._upload(temp_filename, s3_path)
                eprint("=== Finished script to %s ===" % s3_path)
            else:
                eprint("=== Script already exists in S3 ===")
        finally:
            shutil.rmtree(temp_dir)
        
        full_path = "s3://{}/{}".format(self.bucket, s3_path)

        return full_path


def before_run_validations(tracking_uri, cluster_spec):
    """Validations to perform before running a project on Qubole."""
    if cluster_spec is None:
        raise ExecutionException("Cluster spec must be provided when launching MLflow project "
                                 "runs on Qubole.")
    if tracking.utils._is_local_uri(tracking_uri):
        raise ExecutionException(
            "When running on Qubole, the MLflow tracking URI must be "
            "a remote HTTP URI accessible to both the "
            "current client and code running on Qubole. Got local tracking URI %s. "
            "Please specify a valid tracking URI via mlflow.set_tracking_uri or by setting the "
            "MLFLOW_TRACKING_URI environment variable." % tracking_uri)

def _get_qubole_run_script(project_s3_path, run_id, entry_point, parameters, env_vars):
    """
    Generates MLflow CLI command to run on Qubole cluster
    """
    project_dir = QUBOLE_TARFILE_ARCHIVE_NAME
    
    script_template = \
    """
    set -x
    # Activate mlflow environment
    source /usr/lib/envs/mlflow/bin/activate /usr/lib/envs/mlflow/
    # Export environment variables
    {}
    # Untar project
    tar -xf {}
    # Configure boto creds
    source /usr/lib/hustler/bin/qubole-bash-lib.sh
    export AWS_ACCESS_KEY_ID=`nodeinfo s3_access_key_id`
    export AWS_SECRET_ACCESS_KEY=`nodeinfo s3_secret_access_key`
    export HOME=/home/yarn
    # Run mlflow
    mlflow run {} \
    --entry-point {} \
    {} {}
    """

    env_var_export = " && ".join(["export {}={}".format(k, v) for 
                                  (k, v) in env_vars.items()])

    project_tar = project_s3_path.rstrip("/").split("/")[-1]

    run_id_param = "--run-id {}".format(run_id) if run_id else ""
    
    program_params = ""
    if parameters:
        program_params = " ".join(["-P {}={}".format(k, v) for 
                                   (k, v) in parameters.items()])  
    
    mlflow_run_cmd = script_template.format(env_var_export, project_tar,
                                            project_dir, entry_point,
                                            run_id_param, program_params)
    
    return mlflow_run_cmd


def _run_shell_command_job(project_s3_path, script_s3_path, cluster_spec):
    """
    Runs the specified shell command on a Qubole cluster.
    :param project_s3_path: S3 path of archive
    :param script_s3_path: S3 path of shell command to run
    :param cluster_spec: Dictionary describing the cluster, expected to contain the fields for a
    :return: ShellCommand Object.
    """

    args_template = """
                    --script_location {} \\
                    --files {} \\
                    --cluster-label {} \\
                    {} \\
                    --tags {} \\
                    --name {} \\
                    """
    notify = "--notify" if cluster_spec["command"]["notify"] else ""
    args = args_template.format(script_s3_path, project_s3_path, 
                                cluster_spec["cluster"]["label"],
                                notify, ",".join(cluster_spec["command"]["tags"]),
                                cluster_spec["command"]["name"])

    eprint("=== Launching MLflow run as Qubole job ===")
    
    Qubole.configure(**cluster_spec["qubole"])
    args = ShellCommand.parse(shlex.split(args))
    command = ShellCommand.run(**args)

    return command


def run_qubole(remote_run, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec):
    """
    Runs the project at the specified URI on Qubole, returning a `SubmittedRun` that can be
    used to query the run's status or wait for the resulting Qubole command to terminate.
    """
    tracking_uri = tracking.get_tracking_uri()

    with open(cluster_spec, 'r') as handle:
        try:
            cluster_spec = json.load(handle)
        except ValueError:
            eprint("Error when attempting to load and parse JSON cluster spec from file "
                    "%s. " % cluster_spec)
            raise

    project_s3_path = S3Utils(cluster_spec["aws"]).upload_project(work_dir, experiment_id)

    env_vars = {
        tracking._TRACKING_URI_ENV_VAR: tracking_uri,
        tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
        "MLFLOW_CONDA_HOME": QUBOLE_CONDA_HOME
    }

    run_id = remote_run.run_info.run_uuid
    eprint("=== Running entry point %s of project %s on Qubole. ===" % (entry_point, uri))
    
    # Get the shell command to run
    script = _get_qubole_run_script(project_s3_path, run_id, entry_point, parameters, env_vars)
    
    script_s3_path = S3Utils(cluster_spec["aws"]).upload_script(script, experiment_id)

    # Launch run on Qubole  
    command = _run_shell_command_job(project_s3_path, script_s3_path, cluster_spec)
    submitted_run = QuboleSubmittedRun(command, run_id)
    submitted_run._print_description_and_log_tags()
    
    return submitted_run


class QuboleSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Qubole Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.
    """
    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 30

    def __init__(self, cluster_spec, command, run_id):
        super(QuboleSubmittedRun, self).__init__()
        self.cluster_spec = cluster_spec
        self.command = command
        self.run_id = run_id

    def wait(self):
        while not command.is_done(command.status):
            time.sleep(self.POLL_STATUS_INTERVAL)
        return command.is_success(command.status)

    def cancel(self):
        self.command.cancel()
        self.wait()

    def _get_status(self):
        status = self.command.status
        if not self.command.is_done(status):
            return RunStatus.RUNNING
        if self.command.is_success(status):
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_command_url(self):
        qubol_env_base_url = "/".join(self.cluster_spec["qubole"]["api_url"]\
                                          .rstrip("/")\
                                          .split("/")[:-1])       
        command_url = "{}/v2/analyze?command_id={}".format(
                                qubol_env_base_url, self.command.id) 

        return command_url

    def _print_description_and_log_tags(self):
        eprint("=== Launched MLflow run as Qubole command with ID %s. Getting run status "
               "page URL... ===" % self.run_id)
        command_url = self.get_command_url()
        eprint("=== Check the run's status at %s ===" % jobs_page_url)
        tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                        MLFLOW_QUBOLE_COMMAND_URL, command_url)
        tracking.MlflowClient().set_tag(self._mlflow_run_id,
                                        MLFLOW_QUBOLE_COMMAND_ID, self.command.id)

        eprint("=== Check the run's status at %s ===" % command_url)

    def get_status(self):
        return RunStatus.to_string(self._get_status())