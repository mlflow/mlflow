#!/usr/bin/env python
import mock
import numpy
import pytest

from databricks_cli.configure.provider import DatabricksConfig
from mlflow.utils import rest_utils
from mlflow.utils.rest_utils import NumpyEncoder


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_token(get_config):
    get_config.return_value = \
        DatabricksConfig("host", None, None, "mytoken", insecure=False)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params == {
        'hostname': 'host',
        'headers': {
            'Authorization': 'Bearer mytoken'
        },
        'verify': True,
    }


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_user_password(get_config):
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params == {
        'hostname': 'host',
        'headers': {
            'Authorization': 'Basic dXNlcjpwYXNz'
        },
        'verify': True,
    }


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_no_verify(get_config):
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params['verify'] is False


@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_databricks_params_custom_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    params = rest_utils.get_databricks_http_request_kwargs_or_fail("profile")
    assert params['verify'] is False
    ProfileConfigProvider.assert_called_with("profile")


@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_databricks_params_throws_errors(ProfileConfigProvider):
    # No hostname
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig(None, "user", "pass", None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_kwargs_or_fail()

    # No authentication
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig("host", None, None, None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_kwargs_or_fail()


class MockProfileConfigProvider:
    def __init__(self, profile):
        assert profile == "my-profile"

    def get_config(self):
        return DatabricksConfig("host", "user", "pass", None, insecure=False)


import databricks_cli

@mock.patch('requests.request')
@mock.patch('databricks_cli.configure.provider.get_config')
# @mock.patch.object(databricks_cli.configure.provider, 'ProfileConfigProvider', MockProfileConfigProvider)
def test_databricks_http_request_integration(get_config, request):
    """Confirms that the databricks http request params can in fact be used as an HTTP request"""
    def confirm_request_params(**kwargs):
        assert kwargs == {
            'method': 'PUT',
            'url': 'host/api/2.0/clusters/list',
            'headers': {
                'Authorization': 'Basic dXNlcjpwYXNz'
            },
            'verify': True,
            'json': {'a': 'b'}
        }
        http_response = mock.MagicMock()
        http_response.status_code = 200
        http_response.text = '{"OK": "woo"}'
        return http_response
    request.side_effect = confirm_request_params
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)

    response = rest_utils.databricks_api_request('clusters/list', 'PUT',
                                                 json={'a': 'b'})
    assert response == {'OK': 'woo'}
    get_config.reset_mock()

    mock_config_provider = mock.Mock(wraps=MockProfileConfigProvider)
    with mock.patch.object(databricks_cli.configure.provider, 'ProfileConfigProvider',
                           mock_config_provider):
        response = rest_utils.databricks_api_request('clusters/list', 'PUT',
                                                     json={'a': 'b'}, profile="my-profile")
        assert response == {'OK': 'woo'}
        assert get_config.call_count == 0
        assert mock_config_provider.get_config.call_count == 1


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
