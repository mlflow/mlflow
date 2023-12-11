import re
from unittest import mock

import numpy
import pytest
import requests

from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.exceptions import InvalidUrlException, MlflowException, RestException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from mlflow.protos.service_pb2 import GetRun
from mlflow.pyfunc.scoring_server import NumpyEncoder
from mlflow.tracking.request_header.default_request_header_provider import (
    _USER_AGENT,
    DefaultRequestHeaderProvider,
)
from mlflow.utils.rest_utils import (
    MlflowHostCreds,
    _can_parse_as_json_object,
    augmented_raise_for_status,
    call_endpoint,
    call_endpoints,
    http_request,
    http_request_safe,
)

from tests import helper_functions


def test_well_formed_json_error_response():
    with mock.patch(
        "requests.Session.request", return_value=mock.MagicMock(status_code=400, text="{}")
    ):
        host_only = MlflowHostCreds("http://my-host")
        response_proto = GetRun.Response()
        with pytest.raises(RestException, match="INTERNAL_ERROR"):
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


def test_non_json_ok_response():
    with mock.patch(
        "requests.Session.request",
        return_value=mock.MagicMock(status_code=200, text="<html></html>"),
    ):
        host_only = MlflowHostCreds("http://my-host")
        response_proto = GetRun.Response()
        with pytest.raises(
            MlflowException,
            match="API request to endpoint was successful but the response body was not "
            "in a valid JSON format",
        ):
            call_endpoint(host_only, "/api/2.0/fetch-model", "GET", "", response_proto)


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
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


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
        verify=True,
        headers=headers,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_aws_sigv4(request, monkeypatch):
    """This test requires the "requests_auth_aws_sigv4" package to be installed"""

    from requests_auth_aws_sigv4 import AWSSigV4

    monkeypatch.setenvs(
        {
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
    )
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
            verify="/some/path",
            headers={**DefaultRequestHeaderProvider().request_headers(), "test": "header"},
            timeout=120,
        )


@mock.patch("requests.Session.request")
def test_http_request_request_headers_user_agent(request):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(
        PluginRequestHeaderProvider, "in_context", return_value=True
    ), mock.patch.object(
        PluginRequestHeaderProvider,
        "request_headers",
        return_value={_USER_AGENT: "test_user_agent"},
    ):
        host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
        expected_headers = {
            _USER_AGENT: "{} {}".format(
                DefaultRequestHeaderProvider().request_headers()[_USER_AGENT], "test_user_agent"
            )
        }

        response = mock.MagicMock()
        response.status_code = 200
        request.return_value = response
        http_request(host_only, "/my/endpoint", "GET")
        request.assert_called_with(
            "GET",
            "http://my-host/my/endpoint",
            verify="/some/path",
            headers=expected_headers,
            timeout=120,
        )


@mock.patch("requests.Session.request")
def test_http_request_request_headers_user_agent_and_extra_header(request):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(
        PluginRequestHeaderProvider, "in_context", return_value=True
    ), mock.patch.object(
        PluginRequestHeaderProvider,
        "request_headers",
        return_value={_USER_AGENT: "test_user_agent", "header": "value"},
    ):
        host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
        expected_headers = {
            _USER_AGENT: "{} {}".format(
                DefaultRequestHeaderProvider().request_headers()[_USER_AGENT], "test_user_agent"
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
        http_request(host_only, "/my/endpoint", "GET")
        mock_get_http_response_with_retries.assert_called_with(
            mock.ANY,
            mock.ANY,
            5,
            2,
            1.0,
            mock.ANY,
            True,
            headers=mock.ANY,
            verify=mock.ANY,
            timeout=120,
        )
        mock_get_http_response_with_retries.reset_mock()
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "8")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "3")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "1.0")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")
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
