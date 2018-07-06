import os
import time

from databricks_cli.configure import provider

from mlflow.projects import ExecutionException
from mlflow.utils import rest_utils
from mlflow.utils.logging_utils import eprint


def get_databricks_hostname_and_auth():
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


def _get_run_result_state(databricks_run_id):
    """
    Returns the run result state (string) of the Databricks run with the passed-in ID, or None
    if the run is still active. See possible values at
    https://docs.databricks.com/api/latest/jobs.html#runresultstate.
    """
    hostname, token, username, password, = get_databricks_hostname_and_auth()
    auth = (username, password) if username is not None and password is not None else None
    res = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/get", token=token, auth=auth, method="GET",
        params={"run_id": databricks_run_id})
    return res["state"].get("result_state", None)


def wait(databricks_run_id, sleep_interval=30):
    """
    Polls a Databricks Job run (with run ID `databricks_run_id`) for termination, checking the
    run's status every `sleep_interval` seconds.
    """
    result_state = _get_run_result_state(databricks_run_id)
    while result_state is None:
        eprint("=== Databricks run is active, checking run status again after %s seconds "
               "===" % sleep_interval)
        time.sleep(sleep_interval)
        result_state = _get_run_result_state(databricks_run_id)
    if result_state != "SUCCESS":
        raise ExecutionException("=== Databricks run finished with status %s != 'SUCCESS' "
                                 "===" % result_state)
    eprint("=== Run succeeded ===")
