import base64
import time
import logging
import json

import requests

from mlflow import __version__
from mlflow.protos import databricks_pb2
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.string_utils import strip_suffix
from mlflow.exceptions import MlflowException, RestException

RESOURCE_DOES_NOT_EXIST = 'RESOURCE_DOES_NOT_EXIST'

_logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    'User-Agent': 'mlflow-python-client/%s' % __version__
}


def http_request(host_creds, endpoint, retries=3, retry_interval=3,
                 max_rate_limit_interval=60, **kwargs):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Ratelimit
    error code (429) will be retried with an exponential back off (1, 2, 4, ... seconds) for at most
    `max_rate_limit_interval` seconds.  Internal errors (500s) will be retried up to `retries` times
    , waiting `retry_interval` seconds between successive retries. Parses the API response
    (assumed to be JSON) into a Python object and returns it.

    :param host_creds: A :py:class:`mlflow.rest_utils.MlflowHostCreds` object containing
        hostname and optional authentication.
    :return: Parsed API response
    """
    hostname = host_creds.host
    auth_str = None
    if host_creds.username and host_creds.password:
        basic_auth_str = ("%s:%s" % (host_creds.username, host_creds.password)).encode("utf-8")
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    elif host_creds.token:
        auth_str = "Bearer %s" % host_creds.token

    headers = dict(_DEFAULT_HEADERS)
    if auth_str:
        headers['Authorization'] = auth_str

    verify = not host_creds.ignore_tls_verification

    def request_with_ratelimit_retries(max_rate_limit_interval, **kwargs):
        response = requests.request(**kwargs)
        time_left = max_rate_limit_interval
        sleep = 1
        while response.status_code == 429 and time_left > 0:
            _logger.warning(
                "API request to {path} returned status code 429 (Rate limit exceeded). "
                "Retrying in %d seconds. "
                "Will continue to retry 429s for up to %d seconds.",
                sleep, time_left)
            time.sleep(sleep)
            time_left -= sleep
            response = requests.request(**kwargs)
            sleep = min(time_left, sleep*2)  # sleep for 1, 2, 4, ... seconds;
        return response

    cleaned_hostname = strip_suffix(hostname, '/')
    url = "%s%s" % (cleaned_hostname, endpoint)
    for i in range(retries):
        response = request_with_ratelimit_retries(max_rate_limit_interval,
                                                  url=url, headers=headers, verify=verify, **kwargs)
        if response.status_code >= 200 and response.status_code < 500:
            return response
        else:
            _logger.error(
                "API request to %s failed with code %s != 200, retrying up to %s more times. "
                "API response body: %s",
                url, response.status_code, retries - i - 1, response.text)
            time.sleep(retry_interval)
    raise MlflowException("API request to %s failed to return code 200 after %s tries" %
                          (url, retries))


def _can_parse_as_json(string):
    try:
        json.loads(string)
        return True
    except ValueError:
        return False


def http_request_safe(host_creds, endpoint, **kwargs):
    """
    Wrapper around ``http_request`` that also verifies that the request succeeds with code 200.
    """
    response = http_request(host_creds=host_creds, endpoint=endpoint, **kwargs)
    return verify_rest_response(response, endpoint)


def verify_rest_response(response, endpoint):
    """Verify the return code and raise exception if the request was not successful."""
    if response.status_code != 200:
        base_msg = "API request to endpoint %s failed with error code " \
                   "%s != 200" % (endpoint, response.status_code)
        if _can_parse_as_json(response.text):
            raise RestException(json.loads(response.text))
        raise MlflowException("%s. Response body: '%s'" % (base_msg, response.text))
    return response


def _get_path(path_prefix, endpoint_path):
    return "{}{}".format(path_prefix, endpoint_path)


def extract_api_info_for_service(service, path_prefix):
    """ Return a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(path_prefix, endpoint.path)
        res[service().GetRequestClass(service_method)] = (endpoint_path, endpoint.method)
    return res


def call_endpoint(host_creds, endpoint, method, json_body, response_proto):
    # Convert json string to json dictionary, to pass to requests
    if json_body:
        json_body = json.loads(json_body)
    if method == 'GET':
        response = http_request(
            host_creds=host_creds, endpoint=endpoint, method=method, params=json_body)
    else:
        response = http_request(
            host_creds=host_creds, endpoint=endpoint, method=method, json=json_body)
    response = verify_rest_response(response, endpoint)
    js_dict = json.loads(response.text)
    parse_dict(js_dict=js_dict, message=response_proto)
    return response_proto


class MlflowHostCreds(object):
    """
    Provides a hostname and optional authentication for talking to an MLflow tracking server.
    :param host: Hostname (e.g., http://localhost:5000) to MLflow server. Required.
    :param username: Username to use with Basic authentication when talking to server.
        If this is specified, password must also be specified.
    :param password: Password to use with Basic authentication when talking to server.
        If this is specified, username must also be specified.
    :param token: Token to use with Bearer authentication when talking to server.
        If provided, user/password authentication will be ignored.
    :param ignore_tls_verification: If true, we will not verify the server's hostname or TLS
        certificate. This is useful for certain testing situations, but should never be
        true in production.
    """
    def __init__(self, host, username=None, password=None, token=None,
                 ignore_tls_verification=False):
        if not host:
            raise MlflowException("host is a required parameter for MlflowHostCreds")
        self.host = host
        self.username = username
        self.password = password
        self.token = token
        self.ignore_tls_verification = ignore_tls_verification
