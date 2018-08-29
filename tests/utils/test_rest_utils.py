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
