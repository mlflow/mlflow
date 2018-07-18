import json
import time

from six.moves import shlex_quote

from mlflow.entities.source_type import SourceType

from mlflow.projects.pollable_run import DatabricksPollableRun
from mlflow.utils import rest_utils
from mlflow.utils.logging_utils import eprint
from mlflow import tracking
from mlflow.version import VERSION


def _jobs_runs_get(databricks_run_id):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/get", method="GET", params={"run_id": databricks_run_id})


def _jobs_runs_cancel(databricks_run_id):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/cancel", method="POST", req_body_json={"run_id": databricks_run_id})


def _jobs_runs_submit(req_body_json):
    return rest_utils.databricks_api_request(
        endpoint="jobs/runs/submit", method="POST", req_body_json=req_body_json)


def _get_run_result_state(databricks_run_id):
    """
    Returns the run result state (string) of the Databricks run with the passed-in ID, or None
    if the run is still active. See possible values at
    https://docs.databricks.com/api/latest/jobs.html#runresultstate.
    """
    res = _jobs_runs_get(databricks_run_id)
    return res["state"].get("result_state", None)


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


def run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec,
                   git_username, git_password):
    """
    Runs a project on Databricks, returning a `SubmittedRun` that can be used to query the run's
    status or wait for the resulting Databricks Job run to terminate.
    """
    # Create run object with remote tracking server
    tracking_uri = tracking.get_tracking_uri()
    remote_run = _create_databricks_run(
        tracking_uri=tracking_uri, experiment_id=experiment_id, source_name=uri,
        source_version=version, entry_point_name=entry_point)
    # Set up environment variables for remote execution
    env_vars = {"MLFLOW_GIT_URI": uri}
    if git_username is not None:
        env_vars["MLFLOW_GIT_USERNAME"] = git_username
    if git_password is not None:
        env_vars["MLFLOW_GIT_PASSWORD"] = git_password
    if experiment_id is not None:
        env_vars[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    if remote_run is not None:
        env_vars[tracking._TRACKING_URI_ENV_VAR] = tracking.get_tracking_uri()
        env_vars[tracking._RUN_ID_ENV_VAR] = remote_run.run_info.run_uuid
    eprint("=== Running entry point %s of project %s on Databricks. ===" % (entry_point, uri))
    # Launch run on Databricks
    with open(cluster_spec, 'r') as handle:
        cluster_spec = json.load(handle)
    command = _get_databricks_run_cmd(uri, entry_point, version, parameters)
    db_run_id = _run_shell_command_job(uri, command, env_vars, cluster_spec)
    from mlflow.projects.submitted_run import SubmittedRun
    return SubmittedRun(remote_run, DatabricksPollableRun(db_run_id))


def cancel_databricks(databricks_run_id):
    _jobs_runs_cancel(databricks_run_id)


def monitor_databricks(databricks_run_id, sleep_interval=30):
    """
    Polls a Databricks Job run (with run ID `databricks_run_id`) for termination, checking the
    run's status every `sleep_interval` seconds.
    """
    result_state = _get_run_result_state(databricks_run_id)
    while result_state is None:
        time.sleep(sleep_interval)
        result_state = _get_run_result_state(databricks_run_id)
    return result_state == "SUCCESS"
