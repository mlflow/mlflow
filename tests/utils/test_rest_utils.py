import re
import time
import warnings
from unittest import mock

import numpy
import pytest
import requests

from mlflow.deployments.databricks import DatabricksDeploymentClient
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.exceptions import InvalidUrlException, MlflowException, RestException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.protos.service_pb2 import GetRun
from mlflow.pyfunc.scoring_server import NumpyEncoder
from mlflow.tracking.request_header.default_request_header_provider import (
    _CLIENT_VERSION,
    _USER_AGENT,
    DefaultRequestHeaderProvider,
)
from mlflow.utils.rest_utils import (
    _DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING,
    MlflowHostCreds,
    _can_parse_as_json_object,
    augmented_raise_for_status,
    call_endpoint,
    call_endpoints,
    get_workspace_client,
    http_request,
    http_request_safe,
)

from tests import helper_functions


@pytest.mark.parametrize(
    "response_mock",
    [
        helper_functions.create_mock_response(400, "Error message but not a JSON string"),
        helper_functions.create_mock_response(400, ""),
        helper_functions.create_mock_response(400, None),
    ],
)
def test_malformed_json_error_response(response_mock):
    with mock.patch("requests.Session.request", return_value=response_mock):
        host_only = MlflowHostCreds("http://my-host")

        response_proto = GetRun.Response()
        with pytest.raises(
            MlflowException, match="API request to endpoint /my/endpoint failed with error code 400"
        ):
            call_endpoint(host_only, "/my/endpoint", "GET", None, response_proto)


def test_call_endpoints():
    with mock.patch("mlflow.utils.rest_utils.call_endpoint") as mock_call_endpoint:
        response_proto = GetRun.Response()
        mock_call_endpoint.side_effect = [
            RestException({"error_code": ErrorCode.Name(ENDPOINT_NOT_FOUND)}),
            None,
        ]
        host_only = MlflowHostCreds("http://my-host")
        endpoints = [("/my/endpoint", "POST"), ("/my/endpoint", "GET")]
        resp = call_endpoints(host_only, endpoints, "", response_proto)
        mock_call_endpoint.assert_has_calls(
            [
                mock.call(host_only, endpoint, method, "", response_proto, None)
                for endpoint, method in endpoints
            ]
        )
        assert resp is None


def test_call_endpoints_raises_exceptions():
    with mock.patch("mlflow.utils.rest_utils.call_endpoint") as mock_call_endpoint:
        response_proto = GetRun.Response()
        mock_call_endpoint.side_effect = [
            RestException({"error_code": ErrorCode.Name(ENDPOINT_NOT_FOUND)}),
            RestException({"error_code": ErrorCode.Name(ENDPOINT_NOT_FOUND)}),
        ]
        host_only = MlflowHostCreds("http://my-host")
        endpoints = [("/my/endpoint", "POST"), ("/my/endpoint", "GET")]
        with pytest.raises(RestException, match="ENDPOINT_NOT_FOUND"):
            call_endpoints(host_only, endpoints, "", response_proto)
        mock_call_endpoint.side_effect = [RestException({}), None]
        with pytest.raises(RestException, match="INTERNAL_ERROR"):
            call_endpoints(host_only, endpoints, "", response_proto)


@mock.patch("requests.Session.request")
def test_http_request_hostonly(request):
    host_only = MlflowHostCreds("http://my-host")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_cleans_hostname(request):
    # Add a trailing slash, should be removed.
    host_only = MlflowHostCreds("http://my-host/")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_basic_auth(request):
    host_only = MlflowHostCreds("http://my-host", username="user", password="pass")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    headers = DefaultRequestHeaderProvider().request_headers()
    headers["Authorization"] = "Basic dXNlcjpwYXNz"
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        headers=headers,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_aws_sigv4(request, monkeypatch):
    """This test requires the "requests_auth_aws_sigv4" package to be installed"""

    from requests_auth_aws_sigv4 import AWSSigV4

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    aws_sigv4 = MlflowHostCreds("http://my-host", aws_sigv4=True)
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(aws_sigv4, "/my/endpoint", "GET")

    class AuthMatcher:
        def __eq__(self, other):
            return isinstance(other, AWSSigV4)

    request.assert_called_once_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=mock.ANY,
        headers=mock.ANY,
        timeout=mock.ANY,
        auth=AuthMatcher(),
    )


@mock.patch("requests.Session.request")
@mock.patch("mlflow.tracking.request_auth.registry.fetch_auth")
def test_http_request_with_auth(fetch_auth, request):
    mock_fetch_auth = {"test_name": "test_auth_value"}
    fetch_auth.return_value = mock_fetch_auth
    auth = "test_auth_name"
    host_only = MlflowHostCreds("http://my-host", auth=auth)
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")

    fetch_auth.assert_called_with(auth)

    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=mock.ANY,
        headers=mock.ANY,
        timeout=mock.ANY,
        auth=mock_fetch_auth,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_token(request):
    host_only = MlflowHostCreds("http://my-host", token="my-token")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    headers = DefaultRequestHeaderProvider().request_headers()
    headers["Authorization"] = "Bearer my-token"
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        headers=headers,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_insecure(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=False,
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_client_cert_path(request):
    host_only = MlflowHostCreds("http://my-host", client_cert_path="/some/path")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        cert="/some/path",
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_server_cert_path(request):
    host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify="/some/path",
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_content_type_header(request):
    host_only = MlflowHostCreds("http://my-host", token="my-token")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    extra_headers = {"Content-Type": "text/plain"}
    http_request(host_only, "/my/endpoint", "GET", extra_headers=extra_headers)
    headers = DefaultRequestHeaderProvider().request_headers()
    headers["Authorization"] = "Bearer my-token"
    headers["Content-Type"] = "text/plain"
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=True,
        headers=headers,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_request_headers(request):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(PluginRequestHeaderProvider, "in_context", return_value=True):
        host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")

        response = mock.MagicMock()
        response.status_code = 200
        request.return_value = response
        http_request(host_only, "/my/endpoint", "GET")
        request.assert_called_with(
            "GET",
            "http://my-host/my/endpoint",
            allow_redirects=True,
            verify="/some/path",
            headers={**DefaultRequestHeaderProvider().request_headers(), "test": "header"},
            timeout=120,
        )


@mock.patch("requests.Session.request")
def test_http_request_request_headers_default(request):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with (
        mock.patch.object(PluginRequestHeaderProvider, "in_context", return_value=True),
        mock.patch.object(
            PluginRequestHeaderProvider,
            "request_headers",
            return_value={_USER_AGENT: "test_user_agent", _CLIENT_VERSION: "test_client_version"},
        ),
    ):
        host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
        default_headers = DefaultRequestHeaderProvider().request_headers()
        expected_headers = {
            _USER_AGENT: "{} {}".format(default_headers[_USER_AGENT], "test_user_agent"),
            _CLIENT_VERSION: "{} {}".format(
                default_headers[_CLIENT_VERSION], "test_client_version"
            ),
        }

        response = mock.MagicMock()
        response.status_code = 200
        request.return_value = response
        http_request(host_only, "/my/endpoint", "GET")
        request.assert_called_with(
            "GET",
            "http://my-host/my/endpoint",
            allow_redirects=True,
            verify="/some/path",
            headers=expected_headers,
            timeout=120,
        )


@mock.patch("requests.Session.request")
def test_http_request_request_headers_default_and_extra_header(request):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with (
        mock.patch.object(PluginRequestHeaderProvider, "in_context", return_value=True),
        mock.patch.object(
            PluginRequestHeaderProvider,
            "request_headers",
            return_value={
                _USER_AGENT: "test_user_agent",
                _CLIENT_VERSION: "test_client_version",
                "header": "value",
            },
        ),
    ):
        host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
        default_headers = DefaultRequestHeaderProvider().request_headers()
        expected_headers = {
            _USER_AGENT: "{} {}".format(default_headers[_USER_AGENT], "test_user_agent"),
            _CLIENT_VERSION: "{} {}".format(
                default_headers[_CLIENT_VERSION], "test_client_version"
            ),
            "header": "value",
        }

        response = mock.MagicMock()
        response.status_code = 200
        request.return_value = response
        http_request(host_only, "/my/endpoint", "GET")
        request.assert_called_with(
            "GET",
            "http://my-host/my/endpoint",
            allow_redirects=True,
            verify="/some/path",
            headers=expected_headers,
            timeout=120,
        )


def test_http_request_with_invalid_url_raise_invalid_url_exception():
    """InvalidURL exception can be caught by a custom InvalidUrlException"""
    host_only = MlflowHostCreds("http://my-host")

    with pytest.raises(InvalidUrlException, match="Invalid url: http://my-host/invalid_url"):
        with mock.patch("requests.Session.request", side_effect=requests.exceptions.InvalidURL):
            http_request(host_only, "/invalid_url", "GET")


def test_http_request_with_invalid_url_raise_mlflow_exception():
    """The InvalidUrlException can be caught by the MlflowException"""
    host_only = MlflowHostCreds("http://my-host")

    with pytest.raises(MlflowException, match="Invalid url: http://my-host/invalid_url"):
        with mock.patch("requests.Session.request", side_effect=requests.exceptions.InvalidURL):
            http_request(host_only, "/invalid_url", "GET")


def test_ignore_tls_verification_not_server_cert_path():
    with pytest.raises(
        MlflowException,
        match="When 'ignore_tls_verification' is true then 'server_cert_path' must not be set",
    ):
        MlflowHostCreds(
            "http://my-host",
            ignore_tls_verification=True,
            server_cert_path="/some/path",
        )


@mock.patch("requests.Session.request")
def test_http_request_wrapper(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"
    request.return_value = response
    http_request_safe(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=False,
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )
    response.text = "non json"
    request.return_value = response
    http_request_safe(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        allow_redirects=True,
        verify=False,
        headers=DefaultRequestHeaderProvider().request_headers(),
        timeout=120,
    )
    response.status_code = 400
    response.text = ""
    request.return_value = response
    with pytest.raises(MlflowException, match="Response body"):
        http_request_safe(host_only, "/my/endpoint", "GET")
    response.text = (
        '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Node type not supported"}'
    )
    request.return_value = response
    with pytest.raises(RestException, match="RESOURCE_DOES_NOT_EXIST: Node type not supported"):
        http_request_safe(host_only, "/my/endpoint", "GET")


def test_numpy_encoder():
    test_number = numpy.int64(42)
    ne = NumpyEncoder()
    defaulted_val = ne.default(test_number)
    assert defaulted_val == 42


def test_numpy_encoder_fail():
    if not hasattr(numpy, "float128"):
        pytest.skip("numpy on exit this platform has no float128")
    test_number = numpy.float128
    ne = NumpyEncoder()
    with pytest.raises(TypeError, match="not JSON serializable"):
        ne.default(test_number)


def test_can_parse_as_json_object():
    assert _can_parse_as_json_object("{}")
    assert _can_parse_as_json_object('{"a": "b"}')
    assert _can_parse_as_json_object('{"a": {"b": "c"}}')
    assert not _can_parse_as_json_object("[0, 1, 2]")
    assert not _can_parse_as_json_object('"abc"')
    assert not _can_parse_as_json_object("123")


def test_http_request_customize_config(monkeypatch):
    with mock.patch(
        "mlflow.utils.rest_utils._get_http_response_with_retries"
    ) as mock_get_http_response_with_retries:
        host_only = MlflowHostCreds("http://my-host")
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", raising=False)
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
        monkeypatch.delenv("MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER", raising=False)
        http_request(host_only, "/my/endpoint", "GET")
        mock_get_http_response_with_retries.assert_called_with(
            mock.ANY,
            mock.ANY,
            7,
            2,
            1.0,
            mock.ANY,
            True,
            headers=mock.ANY,
            verify=mock.ANY,
            timeout=120,
            respect_retry_after_header=True,
        )
        mock_get_http_response_with_retries.reset_mock()
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "8")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "3")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "1.0")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")
        monkeypatch.setenv("MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER", "false")
        http_request(host_only, "/my/endpoint", "GET")
        mock_get_http_response_with_retries.assert_called_with(
            mock.ANY,
            mock.ANY,
            8,
            3,
            1.0,
            mock.ANY,
            True,
            headers=mock.ANY,
            verify=mock.ANY,
            timeout=300,
            respect_retry_after_header=False,
        )


def test_http_request_explains_how_to_increase_timeout_in_error_message():
    with mock.patch("requests.Session.request", side_effect=requests.exceptions.Timeout):
        with pytest.raises(
            MlflowException,
            match=(
                r"To increase the timeout, set the environment variable "
                + re.escape(str(MLFLOW_HTTP_REQUEST_TIMEOUT))
            ),
        ):
            http_request(MlflowHostCreds("http://my-host"), "/my/endpoint", "GET")


def test_augmented_raise_for_status():
    response = requests.Response()
    response.status_code = 403
    response._content = b"Token expired"

    with mock.patch("requests.Session.request", return_value=response) as mock_request:
        response = requests.get("https://github.com/mlflow/mlflow.git")
        mock_request.assert_called_once()

    with pytest.raises(requests.HTTPError, match="Token expired") as e:
        augmented_raise_for_status(response)

    assert e.value.response == response
    assert e.value.request == response.request
    assert response.text in str(e.value)


def test_provide_redirect_kwarg():
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = http_request(
            MlflowHostCreds("http://my-host"),
            "/my/endpoint",
            "GET",
            allow_redirects=False,
        )

        assert response.text == "mock response"
        mock_request.assert_called_with(
            "GET",
            "http://my-host/my/endpoint",
            allow_redirects=False,
            headers=mock.ANY,
            verify=mock.ANY,
            timeout=120,
        )


def test_http_request_max_retries(monkeypatch):
    monkeypatch.setenv("_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT", "15")
    host_creds = MlflowHostCreds("http://example.com")

    with mock.patch("requests.Session.request") as mock_request:
        with pytest.raises(MlflowException, match="The configured max_retries"):
            http_request(host_creds, "/endpoint", "GET", max_retries=16)
        mock_request.assert_not_called()
        http_request(host_creds, "/endpoint", "GET", max_retries=3)
        mock_request.assert_called_once()


def test_http_request_backoff_factor(monkeypatch):
    monkeypatch.setenv("_MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT", "200")
    host_creds = MlflowHostCreds("http://example.com")

    with mock.patch("requests.Session.request") as mock_request:
        with pytest.raises(MlflowException, match="The configured backoff_factor"):
            http_request(host_creds, "/endpoint", "GET", backoff_factor=250)
        mock_request.assert_not_called()
        http_request(host_creds, "/endpoint", "GET", backoff_factor=10)
        mock_request.assert_called_once()


def test_http_request_negative_max_retries():
    host_creds = MlflowHostCreds("http://example.com")

    with mock.patch("requests.Session.request") as mock_request:
        with pytest.raises(MlflowException, match="The max_retries value must be either"):
            http_request(host_creds, "/endpoint", "GET", max_retries=-1)
        mock_request.assert_not_called()


def test_http_request_negative_backoff_factor():
    host_creds = MlflowHostCreds("http://example.com")

    with mock.patch("requests.Session.request") as mock_request:
        with pytest.raises(MlflowException, match="The backoff_factor value must be"):
            http_request(host_creds, "/endpoint", "GET", backoff_factor=-1)
        mock_request.assert_not_called()


def test_suppress_databricks_retry_after_secs_warnings():
    host_creds = MlflowHostCreds("http://example.com", use_databricks_sdk=True)

    def mock_do(*args, **kwargs):
        warnings.warn(_DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING)
        return mock.MagicMock()

    with (
        warnings.catch_warnings(record=True) as recorded_warnings,
        mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client,
    ):
        warnings.simplefilter("always")
        mock_workspace_client = mock.MagicMock()
        mock_workspace_client.api_client.do = mock_do
        mock_get_workspace_client.return_value = mock_workspace_client
        http_request(host_creds, "/endpoint", "GET")
        mock_get_workspace_client.assert_called_once()
        assert not any(
            _DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING in str(w.message)
            for w in recorded_warnings
        )


def test_databricks_sdk_retry_on_transient_errors():
    """Test that Databricks SDK retries on transient HTTP errors."""
    host_creds = MlflowHostCreds("http://example.com", use_databricks_sdk=True)

    call_count = 0

    def mock_do_failing_then_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # Fail first 2 attempts
            from databricks.sdk.errors import DatabricksError

            from mlflow.protos.databricks_pb2 import ErrorCode

            raise DatabricksError(
                error_code=ErrorCode.Name(ErrorCode.INTERNAL_ERROR), message="Transient error"
            )
        # Success on 3rd attempt
        response_mock = mock.MagicMock()
        response_mock._response = mock.MagicMock()
        return {"contents": response_mock}

    with mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client:
        mock_workspace_client = mock.MagicMock()
        mock_workspace_client.api_client.do = mock_do_failing_then_success
        mock_get_workspace_client.return_value = mock_workspace_client

        # Use smaller retry timeout to make test run faster
        response = http_request(
            host_creds,
            "/endpoint",
            "GET",
            retry_timeout_seconds=10,
            backoff_factor=0.1,  # Very small backoff for faster test
        )

        assert call_count == 3  # Should retry 2 times, succeed on 3rd
        assert response is not None


def test_databricks_sdk_retry_max_retries_exceeded():
    """Test that Databricks SDK stops retrying when max_retries is exceeded."""
    host_creds = MlflowHostCreds("http://example.com", use_databricks_sdk=True)

    call_count = 0

    def mock_do_always_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        from databricks.sdk.errors import DatabricksError

        raise DatabricksError(error_code="INTERNAL_ERROR", message="Always fails")

    with (
        mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client,
        mock.patch("mlflow.utils.rest_utils._logger") as mock_logger,
    ):
        mock_workspace_client = mock.MagicMock()
        mock_workspace_client.api_client.do = mock_do_always_fail
        mock_get_workspace_client.return_value = mock_workspace_client

        response = http_request(host_creds, "/endpoint", "GET", max_retries=3)

        assert call_count == 4  # Initial call + 3 retries
        assert response.status_code == 500  # Should return error response

        # Check that max retries warning was logged
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Max retries (3) exceeded" in warning_call


def test_databricks_sdk_retry_timeout_exceeded():
    """Test that Databricks SDK stops retrying when timeout is exceeded."""
    host_creds = MlflowHostCreds("http://example.com", use_databricks_sdk=True)

    call_count = 0

    def mock_do_always_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        time.sleep(0.1)  # Small delay to ensure timeout
        from databricks.sdk.errors import DatabricksError

        raise DatabricksError(error_code="INTERNAL_ERROR", message="Always fails")

    with (
        mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client,
        mock.patch("mlflow.utils.rest_utils._logger") as mock_logger,
    ):
        mock_workspace_client = mock.MagicMock()
        mock_workspace_client.api_client.do = mock_do_always_fail
        mock_get_workspace_client.return_value = mock_workspace_client

        response = http_request(
            host_creds,
            "/endpoint",
            "GET",
            retry_timeout_seconds=0.2,  # Very short timeout
            max_retries=10,  # High retry limit that shouldn't be reached
        )

        assert call_count >= 1  # At least initial call
        assert response.status_code == 500  # Should return error response

        # Check that timeout warning was logged
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Retry timeout (0.2s) exceeded" in warning_call


def test_databricks_sdk_retry_non_retryable_error():
    """Test that Databricks SDK doesn't retry on non-retryable errors."""
    host_creds = MlflowHostCreds("http://example.com", use_databricks_sdk=True)

    call_count = 0

    def mock_do_non_retryable_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        from databricks.sdk.errors import InvalidParameterValue

        # Use an error code that maps to 400 (non-retryable)
        raise InvalidParameterValue(error_code="INVALID_PARAMETER_VALUE", message="Bad request")

    with mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client:
        mock_workspace_client = mock.MagicMock()
        mock_workspace_client.api_client.do = mock_do_non_retryable_error
        mock_get_workspace_client.return_value = mock_workspace_client

        response = http_request(host_creds, "/endpoint", "GET", max_retries=5)

        assert call_count == 1  # Should not retry on non-retryable error
        assert response.status_code == 400  # Should return 400 for INVALID_PARAMETER_VALUE


def test_databricks_sdk_retry_backoff_calculation():
    """Test that Databricks SDK uses correct exponential backoff timing."""
    from databricks.sdk.errors import DatabricksError

    from mlflow.utils.request_utils import _TRANSIENT_FAILURE_RESPONSE_CODES
    from mlflow.utils.rest_utils import _retry_databricks_sdk_call_with_exponential_backoff

    call_count = 0

    def mock_failing_call():
        nonlocal call_count
        call_count += 1

        raise DatabricksError(error_code="INTERNAL_ERROR", message="Mock error")

    with mock.patch("mlflow.utils.rest_utils._time_sleep") as mock_sleep:
        with pytest.raises(DatabricksError, match="Mock error"):
            _retry_databricks_sdk_call_with_exponential_backoff(
                call_func=mock_failing_call,
                retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES,
                retry_timeout_seconds=10,
                backoff_factor=1,  # Use 1 for predictable calculation
                backoff_jitter=0,  # No jitter for predictable calculation
                max_retries=3,
            )

    # Verify sleep was called with correct intervals
    # attempt 0 (1st retry): 0 seconds (immediate)
    # attempt 1 (2nd retry): 1 * (2^1) = 2 seconds
    # attempt 2 (3rd retry): 1 * (2^2) = 4 seconds
    expected_sleep_times = [0, 2, 4]
    actual_sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
    assert actual_sleep_times == expected_sleep_times
    assert call_count == 4  # Initial + 3 retries


@pytest.mark.skip
def test_timeout_parameter_propagation_with_timeout():
    """Test timeout parameter propagation from http_request to get_workspace_client with timeout."""
    with (
        mock.patch("databricks.sdk.WorkspaceClient") as mock_workspace_client,
        mock.patch("databricks.sdk.config.Config") as mock_config,
    ):
        # Test http_request with timeout via get_workspace_client directly
        mock_workspace_client_instance = mock.MagicMock()
        mock_workspace_client_instance.api_client.do.return_value = {"contents": mock.MagicMock()}
        mock_workspace_client.return_value = mock_workspace_client_instance

        get_workspace_client(
            use_secret_scope_token=False,
            host="http://my-host",
            token=None,
            databricks_auth_profile="my-profile",
            retry_timeout_seconds=None,
            timeout=180,
        )

        mock_config.assert_called_once_with(
            profile="my-profile",
            http_timeout_seconds=180,
            retry_timeout_seconds=mock.ANY,
        )


@pytest.mark.skip
def test_timeout_parameter_propagation_without_timeout():
    """Test timeout param propagation from http_request to get_workspace_client without timeout."""
    with (
        mock.patch("databricks.sdk.WorkspaceClient") as mock_workspace_client,
        mock.patch("databricks.sdk.config.Config") as mock_config,
    ):
        # Test http_request without timeout via get_workspace_client directly
        mock_workspace_client_instance = mock.MagicMock()
        mock_workspace_client_instance.api_client.do.return_value = {"contents": mock.MagicMock()}
        mock_workspace_client.return_value = mock_workspace_client_instance

        get_workspace_client(
            use_secret_scope_token=False,
            host="http://my-host",
            token=None,
            databricks_auth_profile="my-profile",
            retry_timeout_seconds=None,
            timeout=None,
        )

        mock_config.assert_called_once_with(
            profile="my-profile",
            retry_timeout_seconds=mock.ANY,
        )


def test_deployment_client_timeout_propagation(monkeypatch):
    """Test deployment client propagates timeout to workspace client."""

    with (
        mock.patch("mlflow.utils.rest_utils.get_workspace_client") as mock_get_workspace_client,
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds"
        ) as mock_get_databricks_host_creds,
        mock.patch(
            "mlflow.deployments.databricks.get_databricks_host_creds"
        ) as mock_deployment_host_creds,
    ):
        # Mock the host creds to use Databricks SDK
        mock_host_creds = MlflowHostCreds("http://my-host", use_databricks_sdk=True)
        mock_get_databricks_host_creds.return_value = mock_host_creds
        mock_deployment_host_creds.return_value = mock_host_creds

        # Mock workspace client and its response
        mock_workspace_client_instance = mock.MagicMock()
        mock_workspace_client_instance.api_client.do.return_value = {"contents": mock.MagicMock()}
        mock_get_workspace_client.return_value = mock_workspace_client_instance

        # Set the environment variable to a custom value using monkeypatch
        monkeypatch.setenv("MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", "300")

        # Create deployment client and call predict
        client = DatabricksDeploymentClient("databricks")
        client.predict(endpoint="test-endpoint", inputs={"test": "data"})

        # Verify get_workspace_client was called with the deployment predict timeout
        mock_get_workspace_client.assert_called_once_with(
            False,  # use_secret_scope_token
            "http://my-host",  # host
            None,  # token
            None,  # databricks_auth_profile
            retry_timeout_seconds=None,
            timeout=300,  # MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT value
        )
