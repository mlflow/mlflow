import hashlib
import json
import os
import shutil
import tempfile
import textwrap

from databricks_cli.configure import provider
from six.moves import shlex_quote

from mlflow import tracking
from mlflow.version import VERSION
from mlflow.entities.experiment import Experiment
from mlflow.utils import rest_utils, process, file_utils
from mlflow.utils.logging_utils import eprint
from mlflow.projects import ExecutionException, _fetch_project, _get_work_dir, _load_project

DB_CONTAINER_BASE = "/databricks/mlflow"
DB_TARFILE_BASE = os.path.join(DB_CONTAINER_BASE, "project-tars")
DB_PROJECTS_BASE = os.path.join(DB_CONTAINER_BASE, "projects")
DB_TARFILE_ARCHIVE_NAME = "mlflow-project"
DBFS_EXPERIMENT_DIR_BASE = "mlflow-experiments"




def _get_databricks_run_cmd(dbfs_fuse_tar_uri, entry_point, parameters):
    """
    Generates MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks
    """
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
    mkdir -p {0} && mkdir -p {1} &&
    # Rsync to avoid copying archive into local filesystem if it already exists
    rsync -a -v --ignore-existing {2} {0} &&
    # Extract project into a temporary directory
    cd $(mktemp -d) &&
    tar -xzvf {3} &&
    # Atomically move the project into the desired directory
    mv -T {4} {5} &&
    {6}
    """.format(DB_TARFILE_BASE, DB_PROJECTS_BASE, dbfs_fuse_tar_uri, container_tar_path,
               DB_TARFILE_ARCHIVE_NAME, project_dir, mlflow_run_cmd))
    return ["bash", "-c", shell_command]


def _get_db_hostname_and_auth(db_profile):
    """
    Reads the hostname & auth token to use for running on Databricks from the config file created
    by the Databricks CLI.
    """
    config = provider.get_config_for_profile(db_profile)
    if not config.host:
        raise ExecutionException(
            "Databricks CLI configuration for profile '%s' was malformed; no host URL found. "
            "Please configure the Databricks CLI as described in "
            "https://github.com/databricks/databricks-cli" % db_profile)
    return config.host, config.token, config.username, config.password


def _check_databricks_cli_installed():
    cfg_file = os.path.join(os.path.expanduser("~"), ".databrickscfg")
    try:
        process.exec_cmd(["databricks", "--version"])
    except process.ShellCommandException:
        eprint("Could not find Databricks CLI on PATH. Please install and configure the Databricks "
               "CLI as described in https://github.com/databricks/databricks-cli")
        raise
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


def _dbfs_path_exists(dbfs_uri):
    try:
        process.exec_cmd(["databricks", "fs", "ls", dbfs_uri])
        return True
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
        if not _dbfs_path_exists(dbfs_uri):
            _upload_to_dbfs(temp_tar_filename, dbfs_uri, profile)
        else:
            eprint("=== Project already exists in DBFS ===")
        eprint("=== Finished uploading project to %s ===" % dbfs_uri)
    finally:
        shutil.rmtree(temp_tarfile_dir)
    return dbfs_uri


def run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec,
                   db_profile, git_username, git_password):
    _check_databricks_cli_installed()
    databricks_profile = db_profile or provider.DEFAULT_SECTION
    hostname, token, username, password, = _get_db_hostname_and_auth(databricks_profile)
    work_dir = _get_work_dir(uri, use_temp_cwd=False)
    # Fetch the project into work_dir & validate parameters
    _fetch_project(uri, version, work_dir, git_username, git_password)
    project = _load_project(work_dir, uri)
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    # Upload the project to DBFS, get the URI of the project
    final_experiment_id = experiment_id or Experiment.DEFAULT_EXPERIMENT_ID
    dbfs_project_uri = _upload_project_to_dbfs(work_dir, final_experiment_id, databricks_profile)
    env_vars = {"MLFLOW_GIT_URI": uri}
    if tracking._TRACKING_URI_ENV_VAR not in os.environ:
        eprint("Tracking URI is unspecified, data generated for the run (metrics, params,"
               "artifacts) may not be properly persisted. You can configure a tracking URI by "
               "setting the %s environment variable" % tracking._TRACKING_URI_ENV_VAR)
    else:
        env_vars[tracking._TRACKING_URI_ENV_VAR] = os.environ[tracking._TRACKING_URI_ENV_VAR]
    # Pass experiment ID to shell job on Databricks as an environment variable.
    if experiment_id is not None:
        eprint("=== Using experiment ID %s ===" % experiment_id)
        env_vars[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    # Used for testing: if MLFLOW_REMOTE_PIP_URI is set, install (via pip) the package at that URI
    # on the Databricks cluster. Otherwise, just use the current MLflow version.
    mlflow_lib_string = os.environ.get("MLFLOW_REMOTE_PIP_URI", "mlflow==%s" % VERSION)
    # Read cluster spec from file
    with open(cluster_spec, 'r') as handle:
        cluster_spec = json.load(handle)
    fuse_dst_dir = os.path.join("/dbfs/", dbfs_project_uri.strip("dbfs:/"))
    req_body_json = {
        'run_name': 'MLflow Job Run for %s' % uri,
        'new_cluster': cluster_spec,
        'shell_command_task': {
            'command': _get_databricks_run_cmd(fuse_dst_dir, entry_point, parameters),
            "env_vars": env_vars
        },
        "libraries": [{"pypi": {"package": mlflow_lib_string}}]
    }
    # Run on Databricks
    eprint("=== Running entry point %s of project %s on Databricks. ===" % (entry_point, uri))
    auth = (username, password) if username is not None and password is not None else None
    run_submit_res = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/submit", token=token, auth=auth, method="POST",
        req_body_json=req_body_json)
    run_id = run_submit_res["run_id"]
    eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
           "page URL... ===" % run_id)
    run_info = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/get", token=token, auth=auth, method="GET",
        params={"run_id": run_id})
    jobs_page_url = run_info["run_page_url"]
    eprint("=== Check the run's status at %s ===" % jobs_page_url)
