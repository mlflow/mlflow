import base64
import json

import requests

from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)
from mlflow.exceptions import InvalidUrlException, MlflowException, RestException, get_error_code
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
    _TRANSIENT_FAILURE_RESPONSE_CODES,
    _get_http_response_with_retries,
    augmented_raise_for_status,  # noqa: F401
    cloud_storage_http_request,  # noqa: F401
)
from mlflow.utils.string_utils import strip_suffix

RESOURCE_DOES_NOT_EXIST = "RESOURCE_DOES_NOT_EXIST"
_REST_API_PATH_PREFIX = "/api/2.0"


def http_request(
    host_creds,
    endpoint,
    method,
    max_retries=None,
    backoff_factor=None,
    extra_headers=None,
    retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES,
    timeout=None,
    **kwargs,
):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Transient
    errors such as Rate-limited (429), service unavailable (503) and internal error (500) are
    retried with an exponential back off with backoff_factor * (1, 2, 4, ... seconds).
    The function parses the API response (assumed to be JSON) into a Python object and returns it.

    :param host_creds: A :py:class:`mlflow.rest_utils.MlflowHostCreds` object containing
        hostname and optional authentication.
    :param endpoint: a string for service endpoint, e.g. "/path/to/object".
    :param method: a string indicating the method to use, e.g. "GET", "POST", "PUT".
    :param max_retries: maximum number of retries before throwing an exception.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param extra_headers: a dict of HTTP header name-value pairs to be included in the request.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :param timeout: wait for timeout seconds for response from remote server for connect and
      read request.
    :param kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    :return: requests.Response object.
    """
    max_retries = max_retries or MLFLOW_HTTP_REQUEST_MAX_RETRIES.get()
    backoff_factor = backoff_factor or MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.get()
    timeout = timeout or MLFLOW_HTTP_REQUEST_TIMEOUT.get()
    hostname = host_creds.host
    auth_str = None
    if host_creds.username and host_creds.password:
        basic_auth_str = f"{host_creds.username}:{host_creds.password}".encode()
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    elif host_creds.token:
        auth_str = f"Bearer {host_creds.token}"

    from mlflow.tracking.request_header.registry import resolve_request_headers

    headers = dict(**resolve_request_headers())
    if extra_headers:
        headers = dict(**headers, **extra_headers)

    if auth_str:
        headers["Authorization"] = auth_str

    if host_creds.client_cert_path is not None:
        kwargs["cert"] = host_creds.client_cert_path

    if host_creds.aws_sigv4:
        # will overwrite the Authorization header
        from requests_auth_aws_sigv4 import AWSSigV4

        kwargs["auth"] = AWSSigV4("execute-api")

    cleaned_hostname = strip_suffix(hostname, "/")
    url = f"{cleaned_hostname}{endpoint}"
    try:
        return _get_http_response_with_retries(
            method,
            url,
            max_retries,
            backoff_factor,
            retry_codes,
            headers=headers,
            verify=host_creds.verify,
            timeout=timeout,
            **kwargs,
        )
    except requests.exceptions.Timeout as to:
        raise MlflowException(
            f"API request to {url} failed with timeout exception {to}."
            " To increase the timeout, set the environment variable "
            f"{MLFLOW_HTTP_REQUEST_TIMEOUT!s} to a larger value."
        ) from to
    except requests.exceptions.InvalidURL as iu:
        raise InvalidUrlException(f"Invalid url: {url}") from iu
    except Exception as e:
        raise MlflowException(f"API request to {url} failed with exception {e}")


def _can_parse_as_json_object(string):
    try:
        return isinstance(json.loads(string), dict)
    except Exception:
        return False


def http_request_safe(host_creds, endpoint, method, **kwargs):
    """
    Wrapper around ``http_request`` that also verifies that the request succeeds with code 200.
    """
    response = http_request(host_creds=host_creds, endpoint=endpoint, method=method, **kwargs)
    return verify_rest_response(response, endpoint)


def verify_rest_response(response, endpoint):
    """Verify the return code and format, raise exception if the request was not successful."""
    if response.status_code != 200:
        if _can_parse_as_json_object(response.text):
            raise RestException(json.loads(response.text))
        else:
            base_msg = (
                f"API request to endpoint {endpoint} "
                f"failed with error code {response.status_code} != 200"
            )
            raise MlflowException(
                f"{base_msg}. Response body: '{response.text}'",
                error_code=get_error_code(response.status_code),
            )

    # Skip validation for endpoints (e.g. DBFS file-download API) which may return a non-JSON
    # response
    if endpoint.startswith(_REST_API_PATH_PREFIX) and not _can_parse_as_json_object(response.text):
        base_msg = (
            "API request to endpoint was successful but the response body was not "
            "in a valid JSON format"
        )
        raise MlflowException(f"{base_msg}. Response body: '{response.text}'")

    return response


def _get_path(path_prefix, endpoint_path):
    return f"{path_prefix}{endpoint_path}"


def extract_api_info_for_service(service, path_prefix):
    """Return a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(path_prefix, endpoint.path)
        res[service().GetRequestClass(service_method)] = (endpoint_path, endpoint.method)
    return res


def extract_all_api_info_for_service(service, path_prefix):
    """Return a dictionary mapping each API method to a list of tuples [(path, HTTP method)]"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        res[service().GetRequestClass(service_method)] = [
            (_get_path(path_prefix, endpoint.path), endpoint.method) for endpoint in endpoints
        ]
    return res


def call_endpoint(host_creds, endpoint, method, json_body, response_proto, extra_headers=None):
    # Convert json string to json dictionary, to pass to requests
    if json_body:
        json_body = json.loads(json_body)
    call_kwargs = {
        "host_creds": host_creds,
        "endpoint": endpoint,
        "method": method,
    }
    if extra_headers is not None:
        call_kwargs["extra_headers"] = extra_headers
    if method == "GET":
        call_kwargs["params"] = json_body
        response = http_request(**call_kwargs)
    else:
        call_kwargs["json"] = json_body
        response = http_request(**call_kwargs)
    response = verify_rest_response(response, endpoint)
    js_dict = json.loads(response.text)
    parse_dict(js_dict=js_dict, message=response_proto)
    return response_proto


def call_endpoints(host_creds, endpoints, json_body, response_proto, extra_headers=None):
    # The order that the endpoints are called in is defined by the order
    # specified in ModelRegistryService in model_registry.proto
    for i, (endpoint, method) in enumerate(endpoints):
        try:
            return call_endpoint(
                host_creds, endpoint, method, json_body, response_proto, extra_headers
            )
        except RestException as e:
            if e.error_code != ErrorCode.Name(ENDPOINT_NOT_FOUND) or i == len(endpoints) - 1:
                raise e


class MlflowHostCreds:
    """
    Provides a hostname and optional authentication for talking to an MLflow tracking server.
    :param host: Hostname (e.g., http://localhost:5000) to MLflow server. Required.
    :param username: Username to use with Basic authentication when talking to server.
        If this is specified, password must also be specified.
    :param password: Password to use with Basic authentication when talking to server.
        If this is specified, username must also be specified.
    :param token: Token to use with Bearer authentication when talking to server.
        If provided, user/password authentication will be ignored.
    :param aws_sigv4: If true, we will create a signature V4 to be added for any outgoing request.
        Keys for signing the request can be passed via ENV variables,
        or will be fetched via boto3 session.
    :param ignore_tls_verification: If true, we will not verify the server's hostname or TLS
        certificate. This is useful for certain testing situations, but should never be
        true in production.
        If this is set to true ``server_cert_path`` must not be set.
    :param client_cert_path: Path to ssl client cert file (.pem).
        Sets the cert param of the ``requests.request``
        function (see https://requests.readthedocs.io/en/master/api/).
    :param server_cert_path: Path to a CA bundle to use.
        Sets the verify param of the ``requests.request``
        function (see https://requests.readthedocs.io/en/master/api/).
        If this is set ``ignore_tls_verification`` must be false.
    """

    def __init__(
        self,
        host,
        username=None,
        password=None,
        token=None,
        aws_sigv4=False,
        ignore_tls_verification=False,
        client_cert_path=None,
        server_cert_path=None,
    ):
        if not host:
            raise MlflowException(
                message="host is a required parameter for MlflowHostCreds",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if ignore_tls_verification and (server_cert_path is not None):
            raise MlflowException(
                message=(
                    "When 'ignore_tls_verification' is true then 'server_cert_path' "
                    "must not be set! This error may have occurred because the "
                    "'MLFLOW_TRACKING_INSECURE_TLS' and 'MLFLOW_TRACKING_SERVER_CERT_PATH' "
                    "environment variables are both set - only one of these environment "
                    "variables may be set."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.host = host
        self.username = username
        self.password = password
        self.token = token
        self.aws_sigv4 = aws_sigv4
        self.ignore_tls_verification = ignore_tls_verification
        self.client_cert_path = client_cert_path
        self.server_cert_path = server_cert_path

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    @property
    def verify(self):
        if self.server_cert_path is None:
            return not self.ignore_tls_verification
        else:
            return self.server_cert_path
