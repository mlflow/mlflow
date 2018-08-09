#!/usr/bin/env python
import mock
import pytest

from databricks_cli.configure.provider import DatabricksConfig
from mlflow.utils import rest_utils


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


@mock.patch('requests.request')
@mock.patch('databricks_cli.configure.provider.get_config')
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
