import base64
import json
import time

from databricks_cli.configure import provider
import requests

from mlflow.utils.logging_utils import eprint


def get_databricks_hostname_and_auth():
    """
    Reads the hostname & auth token to use for running on Databricks from the config file created
    by the Databricks CLI. Returns a tuple of (hostname, auth, token) to use when making API
    requests.
    """
    profile = provider.DEFAULT_SECTION
    config = provider.get_config_for_profile(profile)
    if config.username is not None and config.password is not None:
        return config.host, (config.username, config.password), config.token
    if config.token:
        return config.host, None, config.token
    raise Exception("Got malformed Databricks CLI profile '%s'. Please make sure the Databricks "
                    "CLI is properly configured as described at "
                    "https://github.com/databricks/databricks-cli." % profile)


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


def http_request(hostname, endpoint, method, auth, headers, req_body_json, params,
                 retries=3, retry_interval=3):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Retries
    up to `retries` times if a request fails with a server error (e.g. error code 500), waiting
    `retry_interval` seconds between successive retries. Parses the API response (assumed to be
    JSON) into a Python object and returns it.

    :param auth: Auth tuple for Basic/Digest/Custom HTTP Auth
    :param headers: Request headers to use when making the HTTP request
    :param req_body_json: Dictionary containing the request body
    :param params: Query parameters for the request
    :return: Parsed API response
    """
    url = "%s%s" % (hostname, endpoint)
    for i in range(retries):
        response = requests.request(method=method, url=url, headers=headers, verify=False,
                                    params=params, json=req_body_json, auth=auth)
        if response.status_code >= 200 and response.status_code < 500:
            return json.loads(response.text)
        else:
            eprint("API request to %s failed with code %s != 200, retrying up to %s more times. "
                   "API response body: %s" % (url, response.status_code, retries - i - 1,
                                              response.text))
            time.sleep(retry_interval)
    raise Exception("API request to %s failed to return code 200 after %s tries" % (url, retries))
