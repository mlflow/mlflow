#!/usr/bin/env python

import mock
import numpy
import pytest

from mlflow.utils.rest_utils import NumpyEncoder, http_request, MlflowHostCreds


@mock.patch('requests.request')
def test_http_request_hostonly(request):
    host_only = MlflowHostCreds("http://my-host")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, '/my/endpoint')
    request.assert_called_with(
        url='http://my-host/my/endpoint',
        verify=True,
        headers={},
    )


@mock.patch('requests.request')
def test_http_request_cleans_hostname(request):
    # Add a trailing slash, should be removed.
    host_only = MlflowHostCreds("http://my-host/")
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, '/my/endpoint')
    request.assert_called_with(
        url='http://my-host/my/endpoint',
        verify=True,
        headers={},
    )


@mock.patch('requests.request')
def test_http_request_with_basic_auth(request):
    host_only = MlflowHostCreds("http://my-host", username='user', password='pass')
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, '/my/endpoint')
    request.assert_called_with(
        url='http://my-host/my/endpoint',
        verify=True,
        headers={
            'Authorization': 'Basic dXNlcjpwYXNz'
        },
    )


@mock.patch('requests.request')
def test_http_request_with_token(request):
    host_only = MlflowHostCreds("http://my-host", token='my-token')
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, '/my/endpoint')
    request.assert_called_with(
        url='http://my-host/my/endpoint',
        verify=True,
        headers={
            'Authorization': 'Bearer my-token'
        },
    )


@mock.patch('requests.request')
def test_http_request_with_insecure(request):
    host_only = MlflowHostCreds("http://my-host", ignore_tls_verification=True)
    response = mock.MagicMock()
    response.status_code = 200
    request.return_value = response
    http_request(host_only, '/my/endpoint')
    request.assert_called_with(
        url='http://my-host/my/endpoint',
        verify=False,
        headers={},
    )


def test_numpy_encoder():
    test_number = numpy.int64(42)
    ne = NumpyEncoder()
    defaulted_val = ne.default(test_number)
    assert defaulted_val is 42


def test_numpy_encoder_fail():
    test_number = numpy.float128
    with pytest.raises(TypeError):
        ne = NumpyEncoder()
        ne.default(test_number)
