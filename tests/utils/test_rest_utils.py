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


def test_databricks_params_token():
    _mock_profile("DEFAULT", DatabricksConfig("host", None, None, "mytoken", insecure=False))
    params = rest_utils.get_databricks_http_request_params_or_fail()
    assert params == {
      'hostname': 'host',
      'headers': {
        'Authorization': 'Basic dG9rZW46bXl0b2tlbg=='
      },
      'secure_verify': True,
    }


def test_databricks_params_user_password():
    _mock_profile("DEFAULT", DatabricksConfig("host", "user", "pass", None, insecure=False))
    params = rest_utils.get_databricks_http_request_params_or_fail()
    assert params == {
      'hostname': 'host',
      'headers': {
        'Authorization': 'Basic dXNlcjpwYXNz'
      },
      'secure_verify': True,
    }


def test_databricks_params_no_verify():
    _mock_profile("DEFAULT", DatabricksConfig("host", "user", "pass", None, insecure=True))
    params = rest_utils.get_databricks_http_request_params_or_fail()
    assert params['secure_verify'] is False


def test_databricks_params_custom_profile():
    _mock_profile("profile", DatabricksConfig("host", "user", "pass", None, insecure=True))
    params = rest_utils.get_databricks_http_request_params_or_fail("profile")
    assert params['secure_verify'] is False


def test_databricks_params_throws_errors():
    _mock_profile("DEFAULT", DatabricksConfig("host", "user", "pass", None, insecure=False))

    # No such profile
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_params_or_fail("profile")

    # No hostname
    _mock_profile("DEFAULT", DatabricksConfig(None, "user", "pass", None, insecure=False))
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_params_or_fail()

    # No authentication
    _mock_profile("DEFAULT", DatabricksConfig("host", None, None, None, insecure=False))
    with pytest.raises(Exception):
        rest_utils.get_databricks_http_request_params_or_fail()


def test_databricks_http_request_integration():
    """Confirms that the databricks http request params can in fact be used as an HTTP request"""
    def confirm_requests(**kvargs):
        assert kvargs == {
            'method': 'PUT',
            'url': 'host/api/2.0/clusters/list',
            'headers': {
                'Authorization': 'Basic dXNlcjpwYXNz'
            },
            'verify': True,
            'params': 'x=y',
            'json': {'a': 'b'}
        }
        response = mock.MagicMock()
        response.status_code = 200
        response.text = '{"OK": "woo"}'
        return response
    requests.request = confirm_requests
    _mock_profile("DEFAULT", DatabricksConfig("host", "user", "pass", None, insecure=False))
    response = rest_utils.databricks_api_request('clusters/list', 'PUT',
                                                 req_body_json={'a': 'b'}, params='x=y')
    assert response == {'OK': 'woo'}
