#!/usr/bin/env python

import mock
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
)
from mlflow.protos.service_pb2 import GetRun
from tests import helper_functions


def test_well_formed_json_error_response():
    with mock.patch("requests.request") as request_mock:
        host_only = MlflowHostCreds("http://my-host")
        response_mock = mock.MagicMock()
        response_mock.status_code = 400
        response_mock.text = "{}"  # well-formed JSON error response
        request_mock.return_value = response_mock

        response_proto = GetRun.Response()
        with pytest.raises(RestException):
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


def test_non_json_ok_response():
    with mock.patch("requests.request") as request_mock:
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
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


@pytest.mark.parametrize(
    "response_mock",
    [
        helper_functions.create_mock_response(400, "Error message but not a JSON string"),
        helper_functions.create_mock_response(400, ""),
        helper_functions.create_mock_response(400, None),
    ],
)
def test_malformed_json_error_response(response_mock):
    with mock.patch("requests.request") as request_mock:
        host_only = MlflowHostCreds("http://my-host")
        request_mock.return_value = response_mock

        response_proto = GetRun.Response()
        with pytest.raises(MlflowException):
            call_endpoint(host_only, "/my/endpoint", "GET", "", response_proto)


@mock.patch("requests.request")
def test_http_request_hostonly(request):
    host_only = MlflowHostCreds("http://my-host")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, headers=_DEFAULT_HEADERS,
    )


@mock.patch("requests.request")
def test_http_request_cleans_hostname(request):
    # Add a trailing slash, should be removed.
    host_only = MlflowHostCreds("http://my-host/")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, headers=_DEFAULT_HEADERS,
    )


@mock.patch("requests.request")
def test_http_request_with_basic_auth(request):
    host_only = MlflowHostCreds("http://my-host", username="user", password="pass")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    headers = dict(_DEFAULT_HEADERS)
    headers["Authorization"] = "Basic dXNlcjpwYXNz"
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, headers=headers,
    )


@mock.patch("requests.request")
def test_http_request_with_token(request):
    host_only = MlflowHostCreds("http://my-host", token="my-token")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    headers = dict(_DEFAULT_HEADERS)
    headers["Authorization"] = "Bearer my-token"
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, headers=headers,
    )


@mock.patch("requests.request")
def test_http_request_with_token_refresh(request):
    host_only = MlflowHostCreds("http://my-host", token="my-token")

    def refresh_token(force_refresh_token=False):  # pylint: disable=unused-argument
        return MlflowHostCreds("http://my-host", token="my-token-2")

    response = mock.MagicMock()
    response.status_code = 401
    request.return_value = response
    http_request(host_only, "/my/endpoint", host_creds_refresh_func=refresh_token)
    headers = dict(_DEFAULT_HEADERS)
    headers["Authorization"] = "Bearer my-token-2"
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, headers=headers,
    )


@mock.patch("requests.request")
def test_http_request_with_insecure(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=False, headers=_DEFAULT_HEADERS,
    )


@mock.patch("requests.request")
def test_http_request_client_cert_path(request):
    host_only = MlflowHostCreds("http://my-host", client_cert_path="/some/path")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=True, cert="/some/path", headers=_DEFAULT_HEADERS,
    )


@mock.patch("requests.request")
def test_http_request_server_cert_path(request):
    host_only = MlflowHostCreds("http://my-host", server_cert_path="/some/path")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify="/some/path", headers=_DEFAULT_HEADERS,
    )


def test_ignore_tls_verification_not_server_cert_path():
    with pytest.raises(MlflowException):
        MlflowHostCreds(
            "http://my-host", ignore_tls_verification=True, server_cert_path="/some/path",
        )


@mock.patch("requests.request")
def test_429_retries(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)

    class MockedResponse(object):
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "mocked text"

    request.side_effect = [MockedResponse(x) for x in (429, 200)]
    assert http_request(host_only, "/my/endpoint", max_rate_limit_interval=0).status_code == 429
    request.side_effect = [MockedResponse(x) for x in (429, 200)]
    assert http_request(host_only, "/my/endpoint", max_rate_limit_interval=1).status_code == 200
    request.side_effect = [MockedResponse(x) for x in (429, 429, 200)]
    assert http_request(host_only, "/my/endpoint", max_rate_limit_interval=1).status_code == 429
    request.side_effect = [MockedResponse(x) for x in (429, 429, 200)]
    assert http_request(host_only, "/my/endpoint", max_rate_limit_interval=2).status_code == 200
    request.side_effect = [MockedResponse(x) for x in (429, 429, 200)]
    assert http_request(host_only, "/my/endpoint", max_rate_limit_interval=3).status_code == 200
    # Test that any non 429 code is returned
    request.side_effect = [MockedResponse(x) for x in (429, 404, 429, 200)]
    assert http_request(host_only, "/my/endpoint").status_code == 404
    # Test that retries work as expected
    request.side_effect = [MockedResponse(x) for x in (429, 503, 429, 200)]
    with pytest.raises(MlflowException, match="failed to return code 200"):
        http_request(host_only, "/my/endpoint", retries=1)
    request.side_effect = [MockedResponse(x) for x in (429, 503, 429, 200)]
    assert http_request(host_only, "/my/endpoint", retries=2).status_code == 200


@mock.patch("requests.request")
def test_http_request_wrapper(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"
    request.return_value = response
    http_request_safe(host_only, "/my/endpoint")
    request.assert_called_with(
        url="http://my-host/my/endpoint", verify=False, headers=_DEFAULT_HEADERS,
    )
    response.status_code = 400
    response.text = ""
    request.return_value = response
    with pytest.raises(MlflowException, match="Response body"):
        http_request_safe(host_only, "/my/endpoint")
    response.text = (
        '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Node type not supported"}'
    )
    request.return_value = response
    with pytest.raises(RestException, match="RESOURCE_DOES_NOT_EXIST: Node type not supported"):
        http_request_safe(host_only, "/my/endpoint")


def test_numpy_encoder():
    test_number = numpy.int64(42)
    ne = NumpyEncoder()
    defaulted_val = ne.default(test_number)
    assert defaulted_val == 42


def test_numpy_encoder_fail():
    if not hasattr(numpy, "float128"):
        pytest.skip("numpy on exit" "this platform has no float128")
    test_number = numpy.float128
    with pytest.raises(TypeError):
        ne = NumpyEncoder()
        ne.default(test_number)
