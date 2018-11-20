import base64
import time
import logging
import json
from json import JSONEncoder

import numpy
import requests

from mlflow.utils.string_utils import strip_suffix
from mlflow.exceptions import MlflowException, RestException


RESOURCE_DOES_NOT_EXIST = 'RESOURCE_DOES_NOT_EXIST'


_logger = logging.getLogger(__name__)


def http_request(host_creds, endpoint, retries=3, retry_interval=3, **kwargs):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Retries
    up to `retries` times if a request fails with a server error (e.g. error code 500), waiting
    `retry_interval` seconds between successive retries. Parses the API response (assumed to be
    JSON) into a Python object and returns it.

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

    headers = {}
    if auth_str:
        headers['Authorization'] = auth_str

    verify = not host_creds.ignore_tls_verification

    cleaned_hostname = strip_suffix(hostname, '/')
    url = "%s%s" % (cleaned_hostname, endpoint)
    for i in range(retries):
        response = requests.request(url=url, headers=headers, verify=verify, **kwargs)
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
    if response.status_code != 200:
        base_msg = "API request to endpoint %s failed with error code " \
                   "%s != 200" % (endpoint, response.status_code)
        if _can_parse_as_json(response.text):
            raise RestException(json.loads(response.text))
        raise MlflowException("%s. Response body: '%s'" % (base_msg, response.text))
    return response


class NumpyEncoder(JSONEncoder):
    """ Special json encoder for numpy types.
    Note that some numpy types doesn't have native python equivalence,
    hence json.dumps will raise TypeError.
    In this case, you'll need to convert your numpy types into its closest python equivalence.
    """

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, numpy.generic):
            return numpy.asscalar(o)
        return JSONEncoder.default(self, o)


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
