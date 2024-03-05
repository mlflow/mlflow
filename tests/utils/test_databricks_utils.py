import builtins
import json
import os
import sys
import time
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import DatabricksConfig
from mlflow.utils import databricks_utils
from mlflow.utils.databricks_utils import (
    check_databricks_secret_scope_access,
    get_databricks_runtime_major_minor_version,
    get_mlflow_credential_context_by_run_id,
    get_workspace_info_from_databricks_secrets,
    get_workspace_info_from_dbutils,
    is_databricks_default_tracking_uri,
    is_running_in_ipython_environment,
)
from mlflow.utils.uri import construct_db_uri_from_profile

from tests.helper_functions import mock_method_chain


def test_no_throw():
    """
    Outside of Databricks the databricks_utils methods should never throw and should only return
    None.
    """
    assert not databricks_utils.is_in_databricks_notebook()
    assert not databricks_utils.is_in_databricks_repo_notebook()
    assert not databricks_utils.is_in_databricks_job()
    assert not databricks_utils.is_dbfs_fuse_available()
    assert not databricks_utils.is_in_databricks_runtime()


@mock.patch("mlflow.utils.databricks_utils.get_config")
def test_databricks_params_token(get_config):
    get_config.return_value = DatabricksConfig.from_token("host", "mytoken", insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == "host"
    assert params.token == "mytoken"
    assert not params.ignore_tls_verification


@mock.patch("mlflow.utils.databricks_utils.get_config")
def test_databricks_params_user_password(get_config):
    get_config.return_value = DatabricksConfig.from_password("host", "user", "pass", insecure=False)
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == "host"
    assert params.username == "user"
    assert params.password == "pass"


@mock.patch("mlflow.utils.databricks_utils.get_config")
def test_databricks_params_no_verify(get_config):
    get_config.return_value = DatabricksConfig.from_password("host", "user", "pass", insecure=True)
    params = databricks_utils.get_databricks_host_creds()
    assert params.ignore_tls_verification


@mock.patch("mlflow.utils.databricks_utils.ProfileConfigProvider")
def test_databricks_params_custom_profile(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", "user", "pass", insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    params = databricks_utils.get_databricks_host_creds(construct_db_uri_from_profile("profile"))
    assert params.ignore_tls_verification
    ProfileConfigProvider.assert_called_with("profile")


@mock.patch("mlflow.utils.databricks_utils.ProfileConfigProvider")
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


@mock.patch("mlflow.utils.databricks_utils.get_config")
def test_databricks_empty_uri(get_config):
    get_config.return_value = None
    with pytest.raises(MlflowException, match="Got malformed Databricks CLI profile"):
        databricks_utils.get_databricks_host_creds("")


@mock.patch("mlflow.utils.databricks_utils.get_config")
def test_databricks_single_slash_in_uri_scheme_throws(get_config):
    get_config.return_value = None
    with pytest.raises(MlflowException, match="URI is formatted incorrectly"):
        databricks_utils.get_databricks_host_creds("databricks:/profile:path")


def test_databricks_model_serving_throws(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    with pytest.raises(MlflowException, match="Unable to read Oauth credentials"):
        databricks_utils.get_databricks_host_creds()


def test_databricks_params_model_serving_oauth_cache(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    monkeypatch.setenv("DATABRICKS_DEPENDENCY_OAUTH_CACHE", "token")
    monkeypatch.setenv("DATABRICKS_DEPENDENCY_OAUTH_CACHE_EXIRY_TS", str(time.time() + 5))
    params = databricks_utils.get_databricks_host_creds()
    assert params.host == "host"
    assert params.token == "token"


@pytest.fixture
def oauth_file(tmp_path):
    token_contents = {"OAUTH_TOKEN": [{"oauthTokenValue": "token2"}]}
    oauth_file = tmp_path.joinpath("model-dependencies-oauth-token")
    with open(oauth_file, "w") as f:
        json.dump(token_contents, f)
    return oauth_file


def test_databricks_params_model_serving_oauth_cache_expired(monkeypatch, oauth_file):
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    monkeypatch.setenv("DATABRICKS_DEPENDENCY_OAUTH_CACHE", "token")
    monkeypatch.setenv("DATABRICKS_DEPENDENCY_OAUTH_CACHE_EXIRY_TS", str(time.time() - 5))
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        # cache should get updated with new token
        assert os.environ["DATABRICKS_DEPENDENCY_OAUTH_CACHE"] == "token2"
        assert float(os.environ["DATABRICKS_DEPENDENCY_OAUTH_CACHE_EXIRY_TS"]) > time.time()
        assert params.host == "host"
        # should use token2 from oauthfile, rather than token from cache
        assert params.token == "token2"


def test_databricks_params_model_serving_read_oauth(monkeypatch, oauth_file):
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        assert os.environ["DATABRICKS_DEPENDENCY_OAUTH_CACHE"] == "token2"
        assert float(os.environ["DATABRICKS_DEPENDENCY_OAUTH_CACHE_EXIRY_TS"]) > time.time()
        assert params.host == "host"
        assert params.token == "token2"


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
    ("tracking_uri", "result"),
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


@mock.patch("mlflow.utils.databricks_utils.ProfileConfigProvider")
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


def test_is_in_databricks_runtime(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "11.x")
    assert databricks_utils.is_in_databricks_runtime()

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION")
    assert not databricks_utils.is_in_databricks_runtime()


def test_is_in_databricks_model_serving_environment(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_ENV", "true")
    assert databricks_utils.is_in_databricks_model_serving_environment()

    monkeypatch.delenv("DATABRICKS_MODEL_SERVING_ENV")
    assert not databricks_utils.is_in_databricks_model_serving_environment()


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

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pyspark":
            return mock_spark
        else:
            return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        assert databricks_utils.get_repl_id() == "testReplId2"


def test_use_repl_context_if_available(tmp_path, monkeypatch):
    # Simulate a case where `dbruntime.databricks_repl_context.get_context` is unavailable.
    with pytest.raises(ModuleNotFoundError, match="No module named 'dbruntime'"):
        from dbruntime.databricks_repl_context import get_context  # noqa: F401

    command_context_mock = mock.MagicMock()
    command_context_mock.jobId().get.return_value = "job_id"
    command_context_mock.tags().get("jobType").get.return_value = "NORMAL"
    with mock.patch(
        "mlflow.utils.databricks_utils._get_command_context", return_value=command_context_mock
    ) as mock_get_command_context:
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_command_context.assert_called_once()

    # Create a fake databricks_repl_context module
    dbruntime = tmp_path.joinpath("dbruntime")
    dbruntime.mkdir()
    dbruntime.joinpath("databricks_repl_context.py").write_text(
        """
def get_context():
    pass
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # Simulate a case where the REPL context object is not initialized.
    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=None,
    ) as mock_get_context, mock.patch(
        "mlflow.utils.databricks_utils._get_command_context", return_value=command_context_mock
    ) as mock_get_command_context:
        assert databricks_utils.get_job_id() == "job_id"
        assert mock_get_command_context.call_count == 1

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(jobId="job_id"),
    ) as mock_get_context, mock.patch("mlflow.utils.databricks_utils._get_dbutils") as mock_dbutils:
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_context.assert_called_once()
        mock_dbutils.assert_not_called()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(notebookId="notebook_id", notebookPath="/Repos/notebook_path"),
    ) as mock_get_context, mock.patch(
        "mlflow.utils.databricks_utils._get_property_from_spark_context"
    ) as mock_spark_context:
        assert databricks_utils.get_notebook_id() == "notebook_id"
        assert databricks_utils.is_in_databricks_repo_notebook()
        assert mock_get_context.call_count == 2
        mock_spark_context.assert_not_called()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(notebookId="notebook_id", notebookPath="/Users/notebook_path"),
    ) as mock_get_context, mock.patch(
        "mlflow.utils.databricks_utils._get_property_from_spark_context"
    ) as mock_spark_context:
        assert not databricks_utils.is_in_databricks_repo_notebook()

    with mock.patch(
        "dbruntime.databricks_repl_context.get_context",
        return_value=mock.MagicMock(isInCluster=True),
    ) as mock_get_context, mock.patch(
        "mlflow.utils._spark_utils._get_active_spark_session"
    ) as mock_spark_session:
        assert databricks_utils.is_in_cluster()
        mock_get_context.assert_called_once()
        mock_spark_session.assert_not_called()


@pytest.mark.parametrize("get_ipython", [True, None])
def test_is_running_in_ipython_environment_works(get_ipython):
    mod_name = "IPython"
    if mod_name in sys.modules:
        ipython_mod = sys.modules.pop(mod_name)
        assert not is_running_in_ipython_environment()
        sys.modules["IPython"] = ipython_mod

        with mock.patch("IPython.get_ipython", return_value=get_ipython):
            assert is_running_in_ipython_environment() == (get_ipython is not None)


def test_get_mlflow_credential_context_by_run_id():
    with mock.patch(
        "mlflow.tracking.artifact_utils.get_artifact_uri", return_value="dbfs:/path/to/artifact"
    ) as mock_get_artifact_uri, mock.patch(
        "mlflow.utils.uri.get_databricks_profile_uri_from_artifact_uri",
        return_value="databricks://path/to/profile",
    ) as mock_get_databricks_profile, mock.patch(
        "mlflow.utils.databricks_utils.MlflowCredentialContext"
    ) as mock_credential_context:
        get_mlflow_credential_context_by_run_id(run_id="abc")
        mock_get_artifact_uri.assert_called_once_with(run_id="abc")
        mock_get_databricks_profile.assert_called_once_with("dbfs:/path/to/artifact")
        mock_credential_context.assert_called_once_with("databricks://path/to/profile")


def test_check_databricks_secret_scope_access():
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.list.return_value = "random"
    with mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils):
        check_databricks_secret_scope_access("scope")
        mock_dbutils.secrets.list.assert_called_once_with("scope")


def test_check_databricks_secret_scope_access_error():
    mock_dbutils = mock.MagicMock()
    mock_dbutils.secrets.list.side_effect = Exception("no scope access")
    with mock.patch(
        "mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils
    ), mock.patch("mlflow.utils.databricks_utils._logger.warning") as mock_warning:
        check_databricks_secret_scope_access("scope")
        mock_warning.assert_called_once_with(
            "Unable to access Databricks secret scope 'scope' for OpenAI credentials that will be "
            "used to deploy the model to Databricks Model Serving. Please verify that the current "
            "Databricks user has 'READ' permission for this scope. For more information, see "
            "https://mlflow.org/docs/latest/python_api/openai/index.html#credential-management-for-openai-on-databricks. "  # noqa: E501
            "Error: no scope access"
        )
        mock_dbutils.secrets.list.assert_called_once_with("scope")


@pytest.mark.parametrize(
    ("version_str", "is_client_image", "major", "minor"),
    [
        ("client.0", True, 0, 0),
        ("client.1", True, 1, 0),
        ("client.1.6", True, 1, 6),
        ("15.1", False, 15, 1),
        ("12.1.1", False, 12, 1),
    ],
)
def test_get_databricks_runtime_major_minor_version(
    monkeypatch, version_str, is_client_image, major, minor
):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", version_str)
    dbr_version = get_databricks_runtime_major_minor_version()

    assert dbr_version.is_client_image == is_client_image
    assert dbr_version.major == major
    assert dbr_version.minor == minor


def test_get_dbr_major_minor_version_throws_on_invalid_version_key(monkeypatch):
    # minor version is not allowed to be a string
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "12.x")
    with pytest.raises(MlflowException, match="Failed to parse databricks runtime version"):
        get_databricks_runtime_major_minor_version()
