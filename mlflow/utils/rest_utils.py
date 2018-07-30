import base64
import json
import time

from databricks_cli.configure import provider
import requests

from mlflow.utils.logging_utils import eprint


def _fail_malformed_databricks_auth(profile):
    raise Exception("Got malformed Databricks CLI profile '%s'. Please make sure the Databricks "
                    "CLI is properly configured as described at "
                    "https://github.com/databricks/databricks-cli." % profile)


def get_databricks_http_request_kwargs_or_fail(profile=None):
    """
    Reads in configuration necessary to make HTTP requests to a Databricks server. This
    uses the Databricks CLI's ConfigProvider interface to load the DatabricksConfig object.
    This method will throw an exception if sufficient auth cannot be found.

    :param profile: Databricks CLI profile. If not provided, we will read the default profile.
    :return: Dictionary with parameters that can be passed to http_request(). This will
             at least include the hostname and headers sufficient to authenticate to Databricks.
    """
    if not profile:
        profile = provider.DEFAULT_SECTION
    config = provider.get_config_for_profile(profile)

    hostname = config.host
    if not hostname:
        _fail_malformed_databricks_auth(profile)

    basic_auth_str = None
    if config.username is not None and config.password is not None:
        basic_auth_str = ("%s:%s" % (config.username, config.password)).encode("utf-8")
    elif config.token:
        basic_auth_str = ("token:%s" % config.token).encode("utf-8")
    if not basic_auth_str:
        _fail_malformed_databricks_auth(profile)

    headers = {
        "Authorization": "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    }

    secure_verify = True
    if hasattr(config, 'insecure') and config.insecure:
        secure_verify = False

    return {
        'hostname': hostname,
        'headers': headers,
        'secure_verify': secure_verify,
    }


def databricks_api_request(endpoint, method, req_body_json=None, params=None):
    final_endpoint = "/api/2.0/%s" % endpoint
    request_params = get_databricks_http_request_kwargs_or_fail()
    return http_request(endpoint=final_endpoint, method=method, req_body_json=req_body_json,
                        params=params, **request_params)


def http_request(hostname, endpoint, method, headers=None, req_body_json=None, params=None,
                 secure_verify=True, retries=3, retry_interval=3):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Retries
    up to `retries` times if a request fails with a server error (e.g. error code 500), waiting
    `retry_interval` seconds between successive retries. Parses the API response (assumed to be
    JSON) into a Python object and returns it.

    :param headers: Request headers to use when making the HTTP request
    :param req_body_json: Dictionary containing the request body
    :param params: Query parameters for the request
    :return: Parsed API response
    """
    url = "%s%s" % (hostname, endpoint)
    for i in range(retries):
        response = requests.request(method=method, url=url, headers=headers, verify=secure_verify,
                                    params=params, json=req_body_json)
        if response.status_code >= 200 and response.status_code < 500:
            return json.loads(response.text)
        else:
            eprint("API request to %s failed with code %s != 200, retrying up to %s more times. "
                   "API response body: %s" % (url, response.status_code, retries - i - 1,
                                              response.text))
            time.sleep(retry_interval)
    raise Exception("API request to %s failed to return code 200 after %s tries" % (url, retries))
