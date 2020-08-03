import mock
import pytest

from mlflow.utils import databricks_utils
from databricks_cli.configure.provider import DatabricksConfig

from mlflow.utils.databricks_utils import get_workspace_info_from_dbutils, \
    get_workspace_info_from_databricks_secrets, is_databricks_default_tracking_uri
from mlflow.utils.uri import construct_db_uri_from_profile


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
    params = databricks_utils.get_databricks_host_creds(construct_db_uri_from_profile("profile"))
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
        params = databricks_utils.get_databricks_host_creds("databricks://profile/prefix")
        mock_dbutils.secrets.get.assert_any_call(key='prefix-host', scope='profile')
        mock_dbutils.secrets.get.assert_any_call(key='prefix-token', scope='profile')
        assert params.host == 'random'
        assert params.token == 'random'


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_empty_uri(get_config):
    get_config.return_value = None
    with pytest.raises(Exception):
        databricks_utils.get_databricks_host_creds("")


@mock.patch('databricks_cli.configure.provider.get_config')
def test_databricks_single_slash_in_uri_scheme_throws(get_config):
    get_config.return_value = None
    with pytest.raises(Exception):
        databricks_utils.get_databricks_host_creds("databricks:/profile/path")


def test_get_workspace_info_from_databricks_secrets():
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.get.return_value = 'workspace-placeholder-info'
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = \
            get_workspace_info_from_databricks_secrets('databricks://profile/prefix')
        mock_dbutils.secrets.get.assert_any_call(key='prefix-host', scope='profile')
        mock_dbutils.secrets.get.assert_any_call(key='prefix-workspace-id', scope='profile')
        assert workspace_host == 'workspace-placeholder-info'
        assert workspace_id == 'workspace-placeholder-info'


def test_get_workspace_info_from_dbutils():
    mock_dbutils = mock.MagicMock()
    mock_dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext\
        .return_value.toJson.return_value = '{"extraContext":' \
                                            '{"api_url": "https://mlflow.databricks.com"},' \
                                            '"tags": {"orgId" : "1111"}}'
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host == 'https://mlflow.databricks.com'
        assert workspace_id == '1111'


def test_get_workspace_info_from_dbutils_when_no_dbutils_available():
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=None):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host is None
        assert workspace_id is None


@pytest.mark.parametrize("tracking_uri, result", [('databricks', True),
                                                  ('databricks://profile/prefix', False),
                                                  ('nondatabricks', False),
                                                  ('databricks\t\r', True), ('databricks\n', True),
                                                  ('databricks://', False),
                                                  ('databricks://aAbB', False)])
def test_is_databricks_default_tracking_uri(tracking_uri, result):
    assert (is_databricks_default_tracking_uri(tracking_uri) == result)


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
