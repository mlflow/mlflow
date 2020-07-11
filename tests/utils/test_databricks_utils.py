import mock
import pytest

from mlflow.utils import databricks_utils
from databricks_cli.configure.provider import DatabricksConfig


def test_no_throw():
    """
    Outside of Databricks the databricks_utils methods should never throw and should only return
    None.
    """
    assert not databricks_utils.is_in_databricks_notebook()
    assert not databricks_utils.is_in_databricks_job()
    assert not databricks_utils.is_dbfs_fuse_available()


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_token(get_config):
    get_config.return_value = \
        DatabricksConfig("host", None, None, "mytoken", insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == 'host'
    assert params.token == 'mytoken'
    assert not params.ignore_tls_verification


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_user_password(get_config):
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == 'host'
    assert params.username == 'user'
    assert params.password == 'pass'


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_params_no_verify(get_config):
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    params = databricks_utils.get_databricks_host_creds()
    assert params.ignore_tls_verification


@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_databricks_params_custom_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    params = databricks_utils.get_databricks_host_creds("profile")
    assert params.ignore_tls_verification
    ProfileConfigProvider.assert_called_with("profile")


@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_databricks_registry_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = None
    ProfileConfigProvider.return_value = mock_provider
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.get.return_value = 'random'
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        params = databricks_utils.get_databricks_host_creds("profile", "prefix")
        mock_dbutils.secrets.get.assert_any_call(key='prefix-host', scope='profile')
        mock_dbutils.secrets.get.assert_any_call(key='prefix-token', scope='profile')
        assert params.host == 'random'
        assert params.token == 'random'


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_empty_uri(get_config):
    get_config.return_value = None
    with pytest.raises(Exception):
        databricks_utils.get_databricks_host_creds("")


@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_databricks_params_throws_errors(ProfileConfigProvider):
    # No hostname
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig(None, "user", "pass", None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception):
        databricks_utils.get_databricks_host_creds()

    # No authentication
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig("host", None, None, None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception):
        databricks_utils.get_databricks_host_creds()
