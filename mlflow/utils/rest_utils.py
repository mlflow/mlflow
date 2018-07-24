import base64
import os
import json

from databricks_cli.configure import provider
import requests

from mlflow.utils.logging_utils import eprint


def get_databricks_hostname_and_auth():
    """
    Reads the hostname & auth token to use for running on Databricks from the config file created
    by the Databricks CLI. Returns a tuple of (hostname, auth, token) to use when making API
    requests.
    """
    home_dir = os.path.expanduser("~")
    cfg_file = os.path.join(home_dir, ".databrickscfg")
    if not os.path.exists(cfg_file):
        raise Exception("Could not find profile for Databricks CLI in %s. Make sure the the "
                        "Databricks CLI is installed and that credentials have been "
                        "configured as described in "
                        "https://github.com/databricks/databricks-cli" % cfg_file)
    else:
        config = provider.get_config_for_profile(provider.DEFAULT_SECTION)
        if config.username is not None and config.password is not None:
            return config.host, (config.username, config.password), config.token
        return config.host, None, config.token


def databricks_api_request(endpoint, method, req_body_json=None, params=None):

    hostname, auth, token = get_databricks_hostname_and_auth()
    final_endpoint = "/api/2.0/%s" % endpoint
    if token is not None:
        token_bytes = ("token:%s" % token).encode("utf-8")
        headers = {
            "Authorization": "Basic " + base64.standard_b64encode(token_bytes).decode("utf-8")
        }
    else:
        headers = None
    # TODO: Remove `verify=False`, currently need it to run against dev shards.
    return http_request(hostname=hostname, endpoint=final_endpoint, method=method, auth=auth,
                        headers=headers, req_body_json=req_body_json, params=params)


def http_request(hostname, endpoint, method, auth, headers, req_body_json, params, retries=3):
    url = "%s%s" % (hostname, endpoint)
    for i in range(retries):
        response = requests.request(method=method, url=url, headers=headers, verify=False,
                                    params=params, json=req_body_json, auth=auth)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            eprint("API request to %s failed with code %s != 200, retrying up to %s more times. "
                   "API response body: %s" % (url, response.status_code, retries - i - 1,
                                              response.text))
    raise Exception("API request to %s failed to return code 200 after %s tries" % (url, retries))
