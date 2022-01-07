#!/usr/bin/env python

from unittest import mock
import numpy
import pytest

from mlflow.exceptions import MlflowException, RestException
from mlflow.pyfunc.scoring_server import NumpyEncoder
from mlflow.utils.rest_utils import (
    http_request,
    http_request_safe,
    MlflowHostCreds,
    _DEFAULT_HEADERS,
    call_endpoint,
    call_endpoints,
    _can_parse_as_json_object,
)
from mlflow.protos.service_pb2 import GetRun
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, ErrorCode
from tests import helper_functions


def test_well_formed_json_error_response():
    with mock.patch("requests.Session.request") as request_mock:
        host_only = MlflowHostCreds("http://my-host")
        response_mock = mock.MagicMock()
        response_mock.status_code = 400
        response_mock.text = "{}"  # well-formed JSON error response
        request_mock.return_value = response_mock

        response_proto = GetRun.Response()
        with pytest.raises(RestException, match="INTERNAL_ERROR"):
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


def test_non_json_ok_response():
    with mock.patch("requests.Session.request") as request_mock:
        host_only = MlflowHostCreds("http://my-host")
        response_mock = mock.MagicMock()
        response_mock.status_code = 200
        response_mock.text = "<html></html>"
        request_mock.return_value = response_mock

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
    with mock.patch("requests.Session.request") as request_mock:
        host_only = MlflowHostCreds("http://my-host")
        request_mock.return_value = response_mock

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
                mock.call(host_only, endpoint, method, "", response_proto)
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
        headers=_DEFAULT_HEADERS,
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
        headers=_DEFAULT_HEADERS,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_basic_auth(request):
    host_only = MlflowHostCreds("http://my-host", username="user", password="pass")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    headers = dict(_DEFAULT_HEADERS)
    headers["Authorization"] = "Basic dXNlcjpwYXNz"
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        verify=True,
        headers=headers,
        timeout=120,
    )


@mock.patch("requests.Session.request")
def test_http_request_with_token(request):
    host_only = MlflowHostCreds("http://my-host", token="my-token")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint", "GET")
    headers = dict(_DEFAULT_HEADERS)
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
        headers=_DEFAULT_HEADERS,
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
        headers=_DEFAULT_HEADERS,
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
        headers=_DEFAULT_HEADERS,
        timeout=120,
    )


@pytest.mark.large
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
            headers={**_DEFAULT_HEADERS, "test": "header"},
            timeout=120,
        )


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
        headers=_DEFAULT_HEADERS,
        timeout=120,
    )
    response.text = "non json"
    request.return_value = response
    http_request_safe(host_only, "/my/endpoint", "GET")
    request.assert_called_with(
        "GET",
        "http://my-host/my/endpoint",
        verify=False,
        headers=_DEFAULT_HEADERS,
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
        pytest.skip("numpy on exit" "this platform has no float128")
    test_number = numpy.float128
    with pytest.raises(TypeError, match="not JSON serializable"):
        ne = NumpyEncoder()
        ne.default(test_number)


def test_can_parse_as_json_object():
    assert _can_parse_as_json_object("{}")
    assert _can_parse_as_json_object('{"a": "b"}')
    assert _can_parse_as_json_object('{"a": {"b": "c"}}')
    assert not _can_parse_as_json_object("[0, 1, 2]")
    assert not _can_parse_as_json_object('"abc"')
    assert not _can_parse_as_json_object("123")
