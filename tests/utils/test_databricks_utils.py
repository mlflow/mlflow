import sys
from unittest import mock
import pytest

from mlflow.utils import databricks_utils
from databricks_cli.configure.provider import DatabricksConfig

from mlflow.exceptions import MlflowException
from mlflow.utils.databricks_utils import (
    get_workspace_info_from_dbutils,
    get_workspace_info_from_databricks_secrets,
    is_databricks_default_tracking_uri,
)
from mlflow.utils.uri import construct_db_uri_from_profile
from tests.helper_functions import mock_method_chain


def test_no_throw():
    """
    Outside of Databricks the databricks_utils methods should never throw and should only return
    None.
    """
    assert not databricks_utils.is_in_databricks_notebook()
    assert not databricks_utils.is_in_databricks_job()
    assert not databricks_utils.is_dbfs_fuse_available()
    assert not databricks_utils.is_in_databricks_runtime()


@mock.patch("databricks_cli.configure.provider.get_config")
def test_databricks_params_token(get_config):
    get_config.return_value = DatabricksConfig.from_token("host", "mytoken", insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == "host"
    assert params.token == "mytoken"
    assert not params.ignore_tls_verification


@mock.patch("databricks_cli.configure.provider.get_config")
def test_databricks_params_user_password(get_config):
    get_config.return_value = DatabricksConfig.from_password("host", "user", "pass", insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == "host"
    assert params.username == "user"
    assert params.password == "pass"


@mock.patch("databricks_cli.configure.provider.get_config")
def test_databricks_params_no_verify(get_config):
    get_config.return_value = DatabricksConfig.from_password("host", "user", "pass", insecure=True)
    params = databricks_utils.get_databricks_host_creds()
    assert params.ignore_tls_verification


@mock.patch("databricks_cli.configure.provider.ProfileConfigProvider")
def test_databricks_params_custom_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", "user", "pass", insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    params = databricks_utils.get_databricks_host_creds(construct_db_uri_from_profile("profile"))
    assert params.ignore_tls_verification
    ProfileConfigProvider.assert_called_with("profile")


@mock.patch("databricks_cli.configure.provider.ProfileConfigProvider")
def test_databricks_registry_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = None
    ProfileConfigProvider.return_value = mock_provider
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.get.return_value = "random"
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        params = databricks_utils.get_databricks_host_creds("databricks://profile:prefix")
        mock_dbutils.secrets.get.assert_any_call(key="prefix-host", scope="profile")
        mock_dbutils.secrets.get.assert_any_call(key="prefix-token", scope="profile")
        assert params.host == "random"
        assert params.token == "random"


@mock.patch("databricks_cli.configure.provider.get_config")
def test_databricks_empty_uri(get_config):
    get_config.return_value = None
    with pytest.raises(MlflowException, match="Got malformed Databricks CLI profile"):
        databricks_utils.get_databricks_host_creds("")


@mock.patch("databricks_cli.configure.provider.get_config")
def test_databricks_single_slash_in_uri_scheme_throws(get_config):
    get_config.return_value = None
    with pytest.raises(MlflowException, match="URI is formatted incorrectly"):
        databricks_utils.get_databricks_host_creds("databricks:/profile:path")


def test_get_workspace_info_from_databricks_secrets():
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.get.return_value = "workspace-placeholder-info"
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = get_workspace_info_from_databricks_secrets(
            "databricks://profile:prefix"
        )
        mock_dbutils.secrets.get.assert_any_call(key="prefix-host", scope="profile")
        mock_dbutils.secrets.get.assert_any_call(key="prefix-workspace-id", scope="profile")
        assert workspace_host == "workspace-placeholder-info"
        assert workspace_id == "workspace-placeholder-info"


def test_get_workspace_info_from_dbutils():
    mock_dbutils = mock.MagicMock()
    methods = ["notebook.entry_point.getDbutils", "notebook", "getContext"]
    mock_method_chain(
        mock_dbutils, methods + ["browserHostName", "get"], return_value="mlflow.databricks.com"
    )
    mock_method_chain(mock_dbutils, methods + ["workspaceId", "get"], return_value="1111")

    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host == "https://mlflow.databricks.com"
        assert workspace_id == "1111"


def test_get_workspace_info_from_dbutils_no_browser_host_name():
    mock_dbutils = mock.MagicMock()
    methods = ["notebook.entry_point.getDbutils", "notebook", "getContext"]
    mock_method_chain(mock_dbutils, methods + ["browserHostName", "get"], return_value=None)
    mock_method_chain(
        mock_dbutils, methods + ["apiUrl", "get"], return_value="https://mlflow.databricks.com"
    )
    mock_method_chain(mock_dbutils, methods + ["workspaceId", "get"], return_value="1111")
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host == "https://mlflow.databricks.com"
        assert workspace_id == "1111"


def test_get_workspace_info_from_dbutils_old_runtimes():
    mock_dbutils = mock.MagicMock()
    methods = ["notebook.entry_point.getDbutils", "notebook", "getContext"]
    mock_method_chain(
        mock_dbutils,
        methods + ["toJson", "get"],
        return_value='{"tags": {"orgId" : "1111", "browserHostName": "mlflow.databricks.com"}}',
    )
    mock_method_chain(
        mock_dbutils, methods + ["browserHostName", "get"], return_value="mlflow.databricks.com"
    )

    # Mock out workspace ID tag
    mock_workspace_id_tag_opt = mock.MagicMock()
    mock_workspace_id_tag_opt.isDefined.return_value = True
    mock_workspace_id_tag_opt.get.return_value = "1111"
    mock_method_chain(
        mock_dbutils, methods + ["tags", "get"], return_value=mock_workspace_id_tag_opt
    )

    # Mimic old runtimes by raising an exception when the nonexistent "workspaceId" method is called
    mock_method_chain(
        mock_dbutils,
        methods + ["workspaceId"],
        side_effect=Exception("workspaceId method not defined!"),
    )
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host == "https://mlflow.databricks.com"
        assert workspace_id == "1111"


def test_get_workspace_info_from_dbutils_when_no_dbutils_available():
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=None):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
        assert workspace_host is None
        assert workspace_id is None


@pytest.mark.parametrize(
    "tracking_uri, result",
    [
        ("databricks", True),
        ("databricks://profile:prefix", False),
        ("databricks://profile/prefix", False),
        ("nondatabricks", False),
        ("databricks\t\r", True),
        ("databricks\n", True),
        ("databricks://", False),
        ("databricks://aAbB", False),
    ],
)
def test_is_databricks_default_tracking_uri(tracking_uri, result):
    assert is_databricks_default_tracking_uri(tracking_uri) == result


@mock.patch("databricks_cli.configure.provider.ProfileConfigProvider")
def test_databricks_params_throws_errors(ProfileConfigProvider):
    # No hostname
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        None, "user", "pass", insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception, match="You haven't configured the CLI yet"):
        databricks_utils.get_databricks_host_creds()

    # No authentication
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", None, None, insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception, match="You haven't configured the CLI yet"):
        databricks_utils.get_databricks_host_creds()


def test_is_in_databricks_runtime():
    with mock.patch(
        "sys.modules",
        new={**sys.modules, "pyspark": mock.MagicMock(), "pyspark.databricks": mock.MagicMock()},
    ):
        # pylint: disable=unused-import,import-error,no-name-in-module,unused-variable
        import pyspark.databricks

        assert databricks_utils.is_in_databricks_runtime()

    with mock.patch("sys.modules", new={**sys.modules, "pyspark": mock.MagicMock()}):
        with pytest.raises(ModuleNotFoundError, match="No module named 'pyspark.databricks'"):
            # pylint: disable=unused-import,import-error,no-name-in-module,unused-variable
            import pyspark.databricks
        assert not databricks_utils.is_in_databricks_runtime()


def test_get_repl_id():
    # Outside of Databricks environments, the Databricks REPL ID should be absent
    assert databricks_utils.get_repl_id() is None

    mock_dbutils = mock.MagicMock()
    mock_dbutils.entry_point.getReplId.return_value = "testReplId1"
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        assert databricks_utils.get_repl_id() == "testReplId1"

    mock_sparkcontext_inst = mock.MagicMock()
    mock_sparkcontext_inst.getLocalProperty.return_value = "testReplId2"
    mock_sparkcontext_class = mock.MagicMock()
    mock_sparkcontext_class.getOrCreate.return_value = mock_sparkcontext_inst
    mock_spark = mock.MagicMock()
    mock_spark.SparkContext = mock_sparkcontext_class

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pyspark":
            return mock_spark
        else:
            return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        assert databricks_utils.get_repl_id() == "testReplId2"


def test_use_repl_context_if_available(tmpdir):
    # Simulate a case where `dbruntime.databricks_repl_context.get_context` is unavailable.
    with pytest.raises(ModuleNotFoundError, match="No module named 'dbruntime'"):
        from dbruntime.databricks_repl_context import get_context  # pylint: disable=unused-import

    command_context_mock = mock.MagicMock()
    command_context_mock.jobId().get.return_value = "job_id"
    with mock.patch(
        "mlflow.utils.databricks_utils._get_command_context", return_value=command_context_mock
    ) as mock_get_command_context:
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_command_context.assert_called_once()

    # Create a fake databricks_repl_context module
    tmpdir.mkdir("dbruntime").join("databricks_repl_context.py").write(
        """
def get_context():
    pass
"""
    )
    sys.path.append(tmpdir.strpath)

    # Simulate a case where the REPL context object is not initialized.
    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=None,
    ) as mock_get_context, mock.patch(
        "mlflow.utils.databricks_utils._get_command_context", return_value=command_context_mock
    ) as mock_get_command_context:
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_command_context.assert_called_once()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(jobId="job_id"),
    ) as mock_get_context, mock.patch("mlflow.utils.databricks_utils._get_dbutils") as mock_dbutils:
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_context.assert_called_once()
        mock_dbutils.assert_not_called()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(notebookId="notebook_id"),
    ) as mock_get_context, mock.patch(
        "mlflow.utils.databricks_utils._get_property_from_spark_context"
    ) as mock_spark_context:
        assert databricks_utils.get_notebook_id() == "notebook_id"
        mock_get_context.assert_called_once()
        mock_spark_context.assert_not_called()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(isInCluster=True),
    ) as mock_get_context, mock.patch(
        "mlflow.utils._spark_utils._get_active_spark_session"
    ) as mock_spark_session:
        assert databricks_utils.is_in_cluster()
        mock_get_context.assert_called_once()
        mock_spark_session.assert_not_called()
