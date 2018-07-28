#!/usr/bin/env python
import mock
import pytest
import requests

from databricks_cli.configure import provider
from databricks_cli.configure.provider import DatabricksConfig
from mlflow.utils import rest_utils


def _mock_profile(expected_profile, config):
    def mock_get_config_for_profile(profile):
        assert profile == expected_profile
        return config
    provider.get_config_for_profile = mock_get_config_for_profile


@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_params_token(get_config_for_profile):
    get_config_for_profile.return_value = \
        DatabricksConfig("host", None, None, "mytoken", insecure=False)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params == {
        'hostname': 'host',
        'headers': {
            'Authorization': 'Basic dG9rZW46bXl0b2tlbg=='
        },
        'secure_verify': True,
    }
    get_config_for_profile.assert_called_with("DEFAULT")


@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_params_user_password(get_config_for_profile):
    get_config_for_profile.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params == {
        'hostname': 'host',
        'headers': {
            'Authorization': 'Basic dXNlcjpwYXNz'
        },
        'secure_verify': True,
    }


@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_params_no_verify(get_config_for_profile):
    get_config_for_profile.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail()
    assert params['secure_verify'] is False


@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_params_custom_profile(get_config_for_profile):
    get_config_for_profile.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    params = rest_utils.get_databricks_http_request_kwargs_or_fail("profile")
    assert params['secure_verify'] is False
    get_config_for_profile.assert_called_with("profile")


@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_params_throws_errors(get_config_for_profile):
    # No hostname
    get_config_for_profile.return_value = \
        DatabricksConfig(None, "user", "pass", None, insecure=False)
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_kwargs_or_fail()

    # No authentication
    get_config_for_profile.return_value = \
        DatabricksConfig("host", None, None, None, insecure=False)
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_kwargs_or_fail()


@mock.patch('requests.request')
@mock.patch('databricks_cli.configure.provider.get_config_for_profile')
def test_databricks_http_request_integration(get_config_for_profile, request):
    """Confirms that the databricks http request params can in fact be used as an HTTP request"""
    def confirm_request_params(**kwargs):
        assert kwargs == {
            'method': 'PUT',
            'url': 'host/api/2.0/clusters/list',
            'headers': {
                'Authorization': 'Basic dXNlcjpwYXNz'
            },
            'verify': True,
            'params': 'x=y',
            'json': {'a': 'b'}
        }
        http_response = mock.MagicMock()
        http_response.status_code = 200
        http_response.text = '{"OK": "woo"}'
        return http_response
    request.side_effect = confirm_request_params
    get_config_for_profile.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)

    response = rest_utils.databricks_api_request('clusters/list', 'PUT',
                                                 req_body_json={'a': 'b'}, params='x=y')
    assert response == {'OK': 'woo'}
