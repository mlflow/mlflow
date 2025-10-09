import base64
import json
import logging
import random
import time
import warnings
from functools import lru_cache
from typing import Any, Callable

import requests

from mlflow.environment_variables import (
    _MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT,
    _MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT,
    MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT,
    MLFLOW_ENABLE_DB_SDK,
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_BACKOFF_JITTER,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
    MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER,
)
from mlflow.exceptions import (
    CUSTOMER_UNAUTHORIZED,
    ERROR_CODE_TO_HTTP_STATUS,
    INVALID_PARAMETER_VALUE,
    InvalidUrlException,
    MlflowException,
    RestException,
    get_error_code,
)
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
    _TRANSIENT_FAILURE_RESPONSE_CODES,
    _get_http_response_with_retries,
    augmented_raise_for_status,  # noqa: F401
    cloud_storage_http_request,  # noqa: F401
)
from mlflow.utils.string_utils import strip_suffix

_logger = logging.getLogger(__name__)

RESOURCE_NON_EXISTENT = "RESOURCE_DOES_NOT_EXIST"
_REST_API_PATH_PREFIX = "/api/2.0"
_UC_OSS_REST_API_PATH_PREFIX = "/api/2.1"
_TRACE_REST_API_PATH_PREFIX = f"{_REST_API_PATH_PREFIX}/mlflow/traces"
_V3_REST_API_PATH_PREFIX = "/api/3.0"
_V3_TRACE_REST_API_PATH_PREFIX = f"{_V3_REST_API_PATH_PREFIX}/mlflow/traces"
_V4_REST_API_PATH_PREFIX = "/api/4.0"
_V4_TRACE_REST_API_PATH_PREFIX = f"{_V4_REST_API_PATH_PREFIX}/mlflow/traces"
_ARMERIA_OK = "200 OK"
_DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING = (
    "The 'retry_after_secs' parameter of DatabricksError is deprecated"
)


def http_request(
    host_creds,
    endpoint,
    method,
    max_retries=None,
    backoff_factor=None,
    backoff_jitter=None,
    extra_headers=None,
    retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES,
    timeout=None,
    raise_on_status=True,
    respect_retry_after_header=None,
    retry_timeout_seconds=None,
    **kwargs,
):
    """Makes an HTTP request with the specified method to the specified hostname/endpoint. Transient
    errors such as Rate-limited (429), service unavailable (503) and internal error (500) are
    retried with an exponential back off with backoff_factor * (1, 2, 4, ... seconds).
    The function parses the API response (assumed to be JSON) into a Python object and returns it.

    Args:
        host_creds: A :py:class:`mlflow.rest_utils.MlflowHostCreds` object containing
            hostname and optional authentication.
        endpoint: A string for service endpoint, e.g. "/path/to/object".
        method: A string indicating the method to use, e.g. "GET", "POST", "PUT".
        max_retries: Maximum number of retries before throwing an exception.
        backoff_factor: A time factor for exponential backoff. e.g. value 5 means the HTTP
            request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
            exponential backoff.
        backoff_jitter: A random jitter to add to the backoff interval.
        extra_headers: A dict of HTTP header name-value pairs to be included in the request.
        retry_codes: A list of HTTP response error codes that qualifies for retry.
        timeout: Wait for timeout seconds for response from remote server for connect and
            read request.
        raise_on_status: Whether to raise an exception, or return a response, if status falls
            in retry_codes range and retries have been exhausted.
        respect_retry_after_header: Whether to respect Retry-After header on status codes defined
            as Retry.RETRY_AFTER_STATUS_CODES or not.
        retry_timeout_seconds: Timeout for retries. Only effective when using Databricks SDK.
        kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    Returns:
        requests.Response object.
    """
    cleaned_hostname = strip_suffix(host_creds.host, "/")
    url = f"{cleaned_hostname}{endpoint}"

    # Set defaults for retry parameters from environment variables if not specified
    max_retries = MLFLOW_HTTP_REQUEST_MAX_RETRIES.get() if max_retries is None else max_retries
    backoff_factor = (
        MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.get() if backoff_factor is None else backoff_factor
    )
    backoff_jitter = (
        MLFLOW_HTTP_REQUEST_BACKOFF_JITTER.get() if backoff_jitter is None else backoff_jitter
    )

    from mlflow.tracking.request_header.registry import resolve_request_headers

    headers = dict(**resolve_request_headers())
    if extra_headers:
        headers = dict(**headers, **extra_headers)

    if host_creds.use_databricks_sdk:
        from databricks.sdk.errors import DatabricksError

        ws_client = get_workspace_client(
            host_creds.use_secret_scope_token,
            host_creds.host,
            host_creds.token,
            host_creds.databricks_auth_profile,
            retry_timeout_seconds=retry_timeout_seconds,
            timeout=timeout,
        )

        def make_sdk_call():
            # Databricks SDK `APIClient.do` API is for making request using
            # HTTP
            # https://github.com/databricks/databricks-sdk-py/blob/a714146d9c155dd1e3567475be78623f72028ee0/databricks/sdk/core.py#L134
            # suppress the warning due to https://github.com/databricks/databricks-sdk-py/issues/963
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=f".*{_DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING}.*"
                )
                raw_response = ws_client.api_client.do(
                    method=method,
                    path=endpoint,
                    headers=headers,
                    raw=True,
                    query=kwargs.get("params"),
                    body=kwargs.get("json"),
                    files=kwargs.get("files"),
                    data=kwargs.get("data"),
                )
                return raw_response["contents"]._response

        try:
            # We retry the SDK call with exponential backoff because the Databricks SDK default
            # retry behavior does not handle all transient errors that we want to retry, and it
            # does not support a customizable retry policy based on HTTP response status codes.
            # Note that, in uncommon cases (due to the limited set if HTTP status codes and
            # response strings that Databricks SDK retries on), the SDK may retry internally,
            # and MLflow may retry on top of that, leading to more retries than specified by
            # `max_retries`. This is acceptable, given the enforcement of an overall request
            # timeout via `retry_timeout_seconds`.
            #
            # TODO: Update transient error handling defaults in Databricks SDK to match standard
            # practices (e.g. retrying on 429, 500, 503, etc.), support custom retries in Databricks
            # SDK, and remove this custom retry wrapper from MLflow
            return _retry_databricks_sdk_call_with_exponential_backoff(
                call_func=make_sdk_call,
                retry_codes=retry_codes,
                retry_timeout_seconds=(
                    retry_timeout_seconds
                    if retry_timeout_seconds is not None
                    else MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT.get()
                ),
                backoff_factor=backoff_factor,
                backoff_jitter=backoff_jitter,
                max_retries=max_retries,
            )
        except DatabricksError as e:
            response = requests.Response()
            response.url = url
            response.status_code = ERROR_CODE_TO_HTTP_STATUS.get(e.error_code, 500)
            response.reason = str(e)
            response.encoding = "UTF-8"
            response._content = json.dumps(
                {
                    "error_code": e.error_code,
                    "message": str(e),
                }
            ).encode("UTF-8")
            return response

    _validate_max_retries(max_retries)
    _validate_backoff_factor(backoff_factor)
    respect_retry_after_header = (
        MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER.get()
        if respect_retry_after_header is None
        else respect_retry_after_header
    )

    timeout = MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout
    auth_str = None
    if host_creds.username and host_creds.password:
        basic_auth_str = f"{host_creds.username}:{host_creds.password}".encode()
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    elif host_creds.token:
        auth_str = f"Bearer {host_creds.token}"
    elif host_creds.client_secret:
        raise MlflowException(
            "To use OAuth authentication, set environmental variable "
            f"'{MLFLOW_ENABLE_DB_SDK.name}' to true",
            error_code=CUSTOMER_UNAUTHORIZED,
        )

    if auth_str:
        headers["Authorization"] = auth_str

    if host_creds.client_cert_path is not None:
        kwargs["cert"] = host_creds.client_cert_path

    if host_creds.aws_sigv4:
        # will overwrite the Authorization header
        from requests_auth_aws_sigv4 import AWSSigV4

        kwargs["auth"] = AWSSigV4("execute-api")
    elif host_creds.auth:
        from mlflow.tracking.request_auth.registry import fetch_auth

        kwargs["auth"] = fetch_auth(host_creds.auth)

    try:
        return _get_http_response_with_retries(
            method,
            url,
            max_retries,
            backoff_factor,
            backoff_jitter,
            retry_codes,
            raise_on_status,
            headers=headers,
            verify=host_creds.verify,
            timeout=timeout,
            respect_retry_after_header=respect_retry_after_header,
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


@lru_cache(maxsize=5)
def get_workspace_client(
    use_secret_scope_token,
    host,
    token,
    databricks_auth_profile,
    retry_timeout_seconds=None,
    timeout=None,
):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    if use_secret_scope_token:
        kwargs = {"host": host, "token": token}
    else:
        kwargs = {"profile": databricks_auth_profile}
    if timeout is not None:
        kwargs["http_timeout_seconds"] = timeout
    config = Config(
        **kwargs,
        retry_timeout_seconds=retry_timeout_seconds
        or MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT.get(),
    )
    # Note: If we use `config` param, all SDK configurations must be set in `config` object.
    return WorkspaceClient(config=config)


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
    # Handle Armeria-specific response case where response text is "200 OK"
    # v1/traces endpoint might return empty response
    if response.status_code == 200 and response.text.strip() in (_ARMERIA_OK, ""):
        response._content = b"{}"  # Update response content to be an empty JSON dictionary
        return response

    # Handle non-200 status codes
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


def _validate_max_retries(max_retries):
    max_retry_limit = _MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT.get()

    if max_retry_limit < 0:
        raise MlflowException(
            message=f"The current maximum retry limit is invalid ({max_retry_limit}). "
            "Cannot be negative.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if max_retries >= max_retry_limit:
        raise MlflowException(
            message=f"The configured max_retries value ({max_retries}) is "
            f"in excess of the maximum allowable retries ({max_retry_limit})",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if max_retries < 0:
        raise MlflowException(
            message=f"The max_retries value must be either 0 a positive integer. Got {max_retries}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_backoff_factor(backoff_factor):
    max_backoff_factor_limit = _MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT.get()

    if max_backoff_factor_limit < 0:
        raise MlflowException(
            message="The current maximum backoff factor limit is invalid "
            f"({max_backoff_factor_limit}). Cannot be negative.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if backoff_factor >= max_backoff_factor_limit:
        raise MlflowException(
            message=f"The configured backoff_factor value ({backoff_factor}) is in excess "
            "of the maximum allowable backoff_factor limit "
            f"({max_backoff_factor_limit})",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if backoff_factor < 0:
        raise MlflowException(
            message="The backoff_factor value must be either 0 a positive integer. "
            f"Got {backoff_factor}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _time_sleep(seconds: float) -> None:
    """
    This function is specifically mocked in `test_rest_utils.py` to test the backoff logic in
    isolation. We avoid wrapping `time.sleep` globally to prevent interfering with unrelated sleep
    calls elsewhere in the codebase or in third-party libraries.
    """
    time.sleep(seconds)


def _retry_databricks_sdk_call_with_exponential_backoff(
    *,
    call_func: Callable[..., Any],
    retry_codes: list[int],
    retry_timeout_seconds: int,
    backoff_factor: int,
    backoff_jitter: float,
    max_retries: int,
):
    """
    Retry a Databricks SDK call with exponential backoff until timeout or max retries reached.

    Args:
        call_func: Function to call that may raise DatabricksError
        retry_codes: Set of HTTP status codes that should trigger retries
        retry_timeout_seconds: Maximum time to spend retrying in seconds
        backoff_factor: Factor for exponential backoff
        backoff_jitter: Random jitter to add to backoff
        max_retries: Maximum number of retry attempts

    Returns:
        The result of call_func() on success

    Raises:
        DatabricksError: If all retries are exhausted or non-retryable error occurs
    """
    from databricks.sdk.errors import STATUS_CODE_MAPPING, DatabricksError

    start_time = time.time()
    attempt = 0

    while attempt <= max_retries:
        try:
            return call_func()
        except DatabricksError as e:
            # Get HTTP status code from the error
            status_code = next(
                (code for code, cls in STATUS_CODE_MAPPING.items() if isinstance(e, cls)), 500
            )
            # Check if this is a retryable error
            if status_code not in retry_codes:
                raise

            # Check if we've exceeded max retries
            if attempt >= max_retries:
                _logger.warning(f"Max retries ({max_retries}) exceeded: {e}")
                raise

            # Calculate backoff time with exponential backoff and jitter
            # NB: Ideally, we'd use urllib3.Retry to compute the jitter, check whether we've
            # exceed max retries, etc. However, urllib3.Retry in urllib3 version 1.x, which MLflow
            # maintains compatibility with, doesn't support retries with jitter
            if attempt <= 0:
                backoff_time = 0  # No backoff on first retry attempt
            else:
                backoff_time = backoff_factor * (2**attempt)
                if backoff_jitter > 0:
                    backoff_time += random.random() * backoff_jitter

            # Check if we've exceeded or would exceed timeout
            elapsed_time = time.time() - start_time
            if elapsed_time + backoff_time >= retry_timeout_seconds:
                _logger.warning(f"Retry timeout ({retry_timeout_seconds}s) exceeded: {e}")
                raise

            _logger.debug(
                f"Databricks SDK call failed with retryable error (status {status_code}): {e}. "
                f"Retrying in {backoff_time:.2f} seconds (attempt {attempt + 1})"
            )

            _time_sleep(backoff_time)
            attempt += 1


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


def get_single_trace_endpoint(request_id, use_v3=True):
    """
    Get the endpoint for a single trace.
    For Databricks tracking URIs, use the V3 API.
    For all other tracking URIs, use the V2 API.

    Args:
        request_id: The trace ID.
        use_v3: Whether to use the V3 API. If True, use the V3 API. If False, use the V2 API.
    """
    if use_v3:
        return f"{_V3_TRACE_REST_API_PATH_PREFIX}/{request_id}"
    return f"{_TRACE_REST_API_PATH_PREFIX}/{request_id}"


def get_single_trace_endpoint_v4(location: str, trace_id: str) -> str:
    """
    Get the endpoint for a single trace using the V4 API.
    """
    return f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location}/{trace_id}"


def get_single_assessment_endpoint_v4(location: str, trace_id: str, assessment_id: str) -> str:
    """
    Get the endpoint for a single assessment using the V4 API.
    """
    return f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location}/{trace_id}/assessments/{assessment_id}"


def get_logged_model_endpoint(model_id: str) -> str:
    return f"{_REST_API_PATH_PREFIX}/mlflow/logged-models/{model_id}"


def get_single_assessment_endpoint(trace_id: str, assessment_id: str) -> str:
    """
    Get the endpoint for a single assessment.

    Args:
        trace_id: The trace ID.
        assessment_id: The assessment ID.
    """
    return f"{_V3_TRACE_REST_API_PATH_PREFIX}/{trace_id}/assessments/{assessment_id}"


def get_trace_tag_endpoint(trace_id):
    """Get the endpoint for trace tags. Always use v2 endpoint."""
    return f"{_REST_API_PATH_PREFIX}/mlflow/traces/{trace_id}/tags"


def call_endpoint(
    host_creds,
    endpoint,
    method,
    json_body,
    response_proto,
    extra_headers=None,
    retry_timeout_seconds=None,
):
    # Convert json string to json dictionary, to pass to requests
    if json_body is not None:
        json_body = json.loads(json_body)
    call_kwargs = {
        "host_creds": host_creds,
        "endpoint": endpoint,
        "method": method,
    }
    if extra_headers is not None:
        call_kwargs["extra_headers"] = extra_headers
    if retry_timeout_seconds is not None:
        call_kwargs["retry_timeout_seconds"] = retry_timeout_seconds
    if method == "GET":
        call_kwargs["params"] = json_body
        response = http_request(**call_kwargs)
    else:
        call_kwargs["json"] = json_body
        response = http_request(**call_kwargs)

    response = verify_rest_response(response, endpoint)
    response_to_parse = response.text
    try:
        js_dict = json.loads(response_to_parse)
    except json.JSONDecodeError:
        if len(response_to_parse) > 50:
            response_to_parse = response_to_parse[:50] + "..."
        _logger.warning(f"Response is not a valid JSON object: {response_to_parse}")
        raise

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

    Args:
        host: Hostname (e.g., http://localhost:5000) to MLflow server. Required.
        username: Username to use with Basic authentication when talking to server.
            If this is specified, password must also be specified.
        password: Password to use with Basic authentication when talking to server.
            If this is specified, username must also be specified.
        token: Token to use with Bearer authentication when talking to server.
            If provided, user/password authentication will be ignored.
        aws_sigv4: If true, we will create a signature V4 to be added for any outgoing request.
            Keys for signing the request can be passed via ENV variables,
            or will be fetched via boto3 session.
        auth: If set, the auth will be added for any outgoing request.
            Keys for signing the request can be passed via ENV variables,
        ignore_tls_verification: If true, we will not verify the server's hostname or TLS
            certificate. This is useful for certain testing situations, but should never be
            true in production.
            If this is set to true ``server_cert_path`` must not be set.
        client_cert_path: Path to ssl client cert file (.pem).
            Sets the cert param of the ``requests.request``
            function (see https://requests.readthedocs.io/en/master/api/).
        server_cert_path: Path to a CA bundle to use.
            Sets the verify param of the ``requests.request``
            function (see https://requests.readthedocs.io/en/master/api/).
            If this is set ``ignore_tls_verification`` must be false.
        use_databricks_sdk: A boolean value represent whether using Databricks SDK for
            authentication.
        databricks_auth_profile: The name of the profile used by Databricks SDK for
            authentication.
        client_id: The client ID used by Databricks OAuth
        client_secret: The client secret used by Databricks OAuth
    """

    def __init__(
        self,
        host,
        username=None,
        password=None,
        token=None,
        aws_sigv4=False,
        auth=None,
        ignore_tls_verification=False,
        client_cert_path=None,
        server_cert_path=None,
        use_databricks_sdk=False,
        databricks_auth_profile=None,
        client_id=None,
        client_secret=None,
        use_secret_scope_token=False,
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
        self.auth = auth
        self.ignore_tls_verification = ignore_tls_verification
        self.client_cert_path = client_cert_path
        self.server_cert_path = server_cert_path
        self.use_databricks_sdk = use_databricks_sdk
        self.databricks_auth_profile = databricks_auth_profile
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_secret_scope_token = use_secret_scope_token

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))

    @property
    def verify(self):
        if self.use_databricks_sdk:
            # Let databricks-sdk set HTTP request `verify` param.
            return None
        if self.server_cert_path is None:
            return not self.ignore_tls_verification
        else:
            return self.server_cert_path
