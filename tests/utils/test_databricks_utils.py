import builtins
import json
import os
import platform
import sys
import time
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
    DatabricksConfig,
    DatabricksModelServingConfigProvider,
)
from mlflow.utils import databricks_utils
from mlflow.utils.databricks_utils import (
    DatabricksConfigProvider,
    DatabricksRuntimeVersion,
    check_databricks_secret_scope_access,
    get_databricks_host_creds,
    get_databricks_runtime_major_minor_version,
    get_dbconnect_udf_sandbox_info,
    get_mlflow_credential_context_by_run_id,
    get_workspace_info_from_databricks_secrets,
    get_workspace_info_from_dbutils,
    get_workspace_url,
    is_databricks_default_tracking_uri,
    is_running_in_ipython_environment,
)
from mlflow.utils.os import is_windows

from tests.helper_functions import mock_method_chain
from tests.pyfunc.test_spark import spark  # noqa: F401


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


def test_databricks_no_creds_found():
    with pytest.raises(MlflowException, match="Reading Databricks credential configuration failed"):
        databricks_utils.get_databricks_host_creds()


def test_databricks_no_creds_found_in_model_serving(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    with pytest.raises(
        MlflowException, match="Reading Databricks credential configuration in model serving failed"
    ):
        databricks_utils.get_databricks_host_creds()


def test_databricks_single_slash_in_uri_scheme_throws():
    with pytest.raises(MlflowException, match="URI is formatted incorrectly"):
        databricks_utils.get_databricks_host_creds("databricks:/profile:path")


@pytest.fixture
def oauth_file(tmp_path):
    token_contents = {"OAUTH_TOKEN": [{"oauthTokenValue": "token2"}]}
    oauth_file = tmp_path.joinpath("model-dependencies-oauth-token")
    with open(oauth_file, "w") as f:
        json.dump(token_contents, f)
    return oauth_file


def test_get_model_dependency_token(oauth_file):
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        token = databricks_utils.get_model_dependency_oauth_token()
        assert token == "token2"


def test_get_model_dependency_oauth_token_model_serving_throws():
    with pytest.raises(MlflowException, match="Unable to read Oauth credentials"):
        databricks_utils.get_model_dependency_oauth_token()


@pytest.mark.parametrize(
    ("model_serving_env_var"),
    [
        ("DATABRICKS_MODEL_SERVING_HOST_URL"),
        ("DB_MODEL_SERVING_HOST_URL"),
    ],
)
def test_databricks_params_model_serving_oauth_cache_databricks(
    monkeypatch, oauth_file, model_serving_env_var
):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv(model_serving_env_var, "host")
    monkeypatch.setenv("DB_DEPENDENCY_OAUTH_CACHE", "token")
    monkeypatch.setenv("DB_DEPENDENCY_OAUTH_CACHE_EXPIRY_TS", str(time.time() + 5))
    # oauth file still needs to be present for should_fetch_model_serving_environment_oauth()
    #  to evaluate true
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        assert params.host == "host"
        # should use token from cache, rather than token from oauthfile
        assert params.token == "token"


def test_databricks_params_model_serving_oauth_cache_expired(monkeypatch, oauth_file):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    monkeypatch.setenv("DB_DEPENDENCY_OAUTH_CACHE", "token")
    monkeypatch.setenv("DB_DEPENDENCY_OAUTH_CACHE_EXPIRY_TS", str(time.time() - 5))
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        # cache should get updated with new token
        assert os.environ["DB_DEPENDENCY_OAUTH_CACHE"] == "token2"
        assert float(os.environ["DB_DEPENDENCY_OAUTH_CACHE_EXPIRY_TS"]) > time.time()
        assert params.host == "host"
        # should use token2 from oauthfile, rather than token from cache
        assert params.token == "token2"


def test_databricks_params_model_serving_read_oauth(monkeypatch, oauth_file):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        assert os.environ["DB_DEPENDENCY_OAUTH_CACHE"] == "token2"
        assert float(os.environ["DB_DEPENDENCY_OAUTH_CACHE_EXPIRY_TS"]) > time.time()
        assert params.host == "host"
        assert params.token == "token2"


def test_databricks_params_env_var_overrides_model_serving_oauth(monkeypatch, oauth_file):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_MODEL_SERVING_HOST_URL", "host")
    monkeypatch.setenv("DATABRICKS_HOST", "host_envvar")
    monkeypatch.setenv("DATABRICKS_TOKEN", "pat_token")
    # oauth file still needs to be present for should_fetch_model_serving_environment_oauth()
    #  to evaluate true
    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        params = databricks_utils.get_databricks_host_creds()
        # should use token and host from envvar, rather than token from oauthfile
        assert params.use_databricks_sdk


def test_model_serving_config_provider_errors_caught():
    provider = DatabricksModelServingConfigProvider()
    with mock.patch.object(
        provider,
        "_get_databricks_model_serving_config",
        side_effect=Exception("Failed to Read OAuth Creds"),
    ):
        assert provider.get_config() is None


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
    with pytest.raises(Exception, match="Reading Databricks credential configuration failed with"):
        databricks_utils.get_databricks_host_creds()

    # No authentication
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", None, None, insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    with pytest.raises(Exception, match="Reading Databricks credential configuration failed with"):
        databricks_utils.get_databricks_host_creds()


def test_is_in_databricks_runtime(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "11.x")
    assert databricks_utils.is_in_databricks_runtime()

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION")
    assert not databricks_utils.is_in_databricks_runtime()


def test_is_in_databricks_model_serving_environment(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    assert databricks_utils.is_in_databricks_model_serving_environment()

    monkeypatch.delenv("IS_IN_DB_MODEL_SERVING_ENV")
    assert not databricks_utils.is_in_databricks_model_serving_environment()

    # Backward compatibility with old env var name
    monkeypatch.setenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", "true")
    assert databricks_utils.is_in_databricks_model_serving_environment()


# test both is_in_databricks_model_serving_environment and
# should_fetch_model_serving_environment_oauth return apprropriate values
def test_should_fetch_model_serving_environment_oauth(monkeypatch, oauth_file):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    # will return false if file mount is not configured even if env var set
    assert not databricks_utils.should_fetch_model_serving_environment_oauth()

    with mock.patch(
        "mlflow.utils.databricks_utils._MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH", str(oauth_file)
    ):
        # both file mount and env var exist, both values should return true
        assert databricks_utils.should_fetch_model_serving_environment_oauth()

        # file mount without env var should return false
        monkeypatch.delenv("IS_IN_DB_MODEL_SERVING_ENV")
        assert not databricks_utils.should_fetch_model_serving_environment_oauth()


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
    with (
        mock.patch(
            "dbruntime.databricks_repl_context.get_context",
            return_value=None,
        ) as mock_get_context,
        mock.patch(
            "mlflow.utils.databricks_utils._get_command_context", return_value=command_context_mock
        ) as mock_get_command_context,
    ):
        assert databricks_utils.get_job_id() == "job_id"
        assert mock_get_command_context.call_count == 1

    with (
        mock.patch(
            "dbruntime.databricks_repl_context.get_context",
            return_value=mock.MagicMock(jobId="job_id"),
        ) as mock_get_context,
        mock.patch("mlflow.utils.databricks_utils._get_dbutils") as mock_dbutils,
    ):
        assert databricks_utils.get_job_id() == "job_id"
        mock_get_context.assert_called_once()
        mock_dbutils.assert_not_called()

    with (
        mock.patch(
            "dbruntime.databricks_repl_context.get_context",
            return_value=mock.MagicMock(
                notebookId="notebook_id", notebookPath="/Repos/notebook_path"
            ),
        ) as mock_get_context,
        mock.patch(
            "mlflow.utils.databricks_utils._get_property_from_spark_context"
        ) as mock_spark_context,
    ):
        assert databricks_utils.get_notebook_id() == "notebook_id"
        assert databricks_utils.is_in_databricks_repo_notebook()
        assert mock_get_context.call_count == 2
        mock_spark_context.assert_not_called()

    with (
        mock.patch(
            "dbruntime.databricks_repl_context.get_context",
            return_value=mock.MagicMock(
                notebookId="notebook_id", notebookPath="/Users/notebook_path"
            ),
        ) as mock_get_context,
        mock.patch(
            "mlflow.utils.databricks_utils._get_property_from_spark_context"
        ) as mock_spark_context,
    ):
        assert not databricks_utils.is_in_databricks_repo_notebook()

    with (
        mock.patch(
            "dbruntime.databricks_repl_context.get_context",
            return_value=mock.MagicMock(isInCluster=True),
        ) as mock_get_context,
        mock.patch("mlflow.utils._spark_utils._get_active_spark_session") as mock_spark_session,
    ):
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
    with (
        mock.patch(
            "mlflow.tracking.artifact_utils.get_artifact_uri", return_value="dbfs:/path/to/artifact"
        ) as mock_get_artifact_uri,
        mock.patch(
            "mlflow.utils.uri.get_databricks_profile_uri_from_artifact_uri",
            return_value="databricks://path/to/profile",
        ) as mock_get_databricks_profile,
        mock.patch(
            "mlflow.utils.databricks_utils.MlflowCredentialContext"
        ) as mock_credential_context,
    ):
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
    with (
        mock.patch("mlflow.utils.databricks_utils._get_dbutils", return_value=mock_dbutils),
        mock.patch("mlflow.utils.databricks_utils._logger.warning") as mock_warning,
    ):
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


def test_prioritize_env_var_config_provider(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "my_host1")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token1")

    class MyProvider(DatabricksConfigProvider):
        def get_config(self):
            return DatabricksConfig(host="my_host2", token="token2")

    monkeypatch.setattr(databricks_utils, "_dynamic_token_config_provider", MyProvider)

    hc = get_databricks_host_creds("databricks")
    assert hc.host == "my_host1"
    assert hc.token == "token1"


@pytest.mark.parametrize(
    ("input_url", "expected_result"),
    [
        # Test with a valid URL without https:// prefix
        ("example.com", "https://example.com"),
        # Test with a valid URL with https:// prefix
        ("https://example.com", "https://example.com"),
        # Test with None URL
        (None, None),
    ],
)
def test_get_workspace_url(input_url, expected_result):
    with mock.patch("mlflow.utils.databricks_utils._get_workspace_url", return_value=input_url):
        result = get_workspace_url()
        assert result == expected_result


@pytest.mark.skipif(is_windows(), reason="This test doesn't work on Windows")
def test_get_dbconnect_udf_sandbox_info(spark, monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "client.1.2")
    databricks_utils._dbconnect_udf_sandbox_info_cache = None

    spark.udf.register(
        "current_version",
        lambda: {"dbr_version": "15.4.x-scala2.12"},
        returnType="dbr_version string",
    )

    info = get_dbconnect_udf_sandbox_info(spark)
    assert info.mlflow_version == mlflow.__version__
    assert info.image_version == "client.1.2"
    assert info.runtime_version == "15.4"
    assert info.platform_machine == platform.machine()

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION")
    databricks_utils._dbconnect_udf_sandbox_info_cache = None

    info = get_dbconnect_udf_sandbox_info(spark)
    assert info.mlflow_version == mlflow.__version__
    assert info.image_version == "15.4"
    assert info.runtime_version == "15.4"
    assert info.platform_machine == platform.machine()


def test_construct_databricks_uc_registered_model_url():
    # Test case with workspace ID
    workspace_url = "https://databricks.com"
    registered_model_name = "name.mlflow.echo_model"
    version = "6"
    workspace_id = "123"

    expected_url = (
        "https://databricks.com/explore/data/models/name/mlflow/echo_model/version/6?o=123"
    )

    result = databricks_utils._construct_databricks_uc_registered_model_url(
        workspace_url=workspace_url,
        registered_model_name=registered_model_name,
        version=version,
        workspace_id=workspace_id,
    )

    assert result == expected_url

    # Test case without workspace ID
    expected_url_no_workspace = (
        "https://databricks.com/explore/data/models/name/mlflow/echo_model/version/6"
    )

    result_no_workspace = databricks_utils._construct_databricks_uc_registered_model_url(
        workspace_url=workspace_url,
        registered_model_name=registered_model_name,
        version=version,
    )

    assert result_no_workspace == expected_url_no_workspace


def test_construct_databricks_logged_model_url():
    # Test case with workspace ID
    workspace_url = "https://databricks.com"
    experiment_id = "123456"
    model_id = "model_789"
    workspace_id = "123"

    expected_url = "https://databricks.com/ml/experiments/123456/models/model_789?o=123"

    result = databricks_utils._construct_databricks_logged_model_url(
        workspace_url=workspace_url,
        experiment_id=experiment_id,
        model_id=model_id,
        workspace_id=workspace_id,
    )

    assert result == expected_url

    # Test case without workspace ID
    expected_url_no_workspace = "https://databricks.com/ml/experiments/123456/models/model_789"

    result_no_workspace = databricks_utils._construct_databricks_logged_model_url(
        workspace_url=workspace_url,
        experiment_id=experiment_id,
        model_id=model_id,
    )

    assert result_no_workspace == expected_url_no_workspace


def test_print_databricks_deployment_job_url():
    workspace_url = "https://databricks.com"
    job_id = "123"
    workspace_id = "456"

    expected_url_no_workspace = "https://databricks.com/jobs/123"
    expected_url = f"{expected_url_no_workspace}?o=456"
    model_name = "main.models.name"

    with (
        mock.patch("mlflow.utils.databricks_utils.eprint") as mock_eprint,
        mock.patch("mlflow.utils.databricks_utils.get_workspace_url", return_value=workspace_url),
    ):
        # Test case with a workspace ID
        with mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id", return_value=workspace_id
        ):
            result = databricks_utils._print_databricks_deployment_job_url(
                model_name=model_name,
                job_id=job_id,
            )

            assert result == expected_url
            mock_eprint.assert_called_once_with(
                f"🔗 Linked deployment job to '{model_name}': {expected_url}"
            )
            mock_eprint.reset_mock()

        # Test case without a workspace ID
        with mock.patch("mlflow.utils.databricks_utils.get_workspace_id", return_value=None):
            result_no_workspace = databricks_utils._print_databricks_deployment_job_url(
                model_name=model_name,
                job_id=job_id,
            )

            assert result_no_workspace == expected_url_no_workspace
            mock_eprint.assert_called_once_with(
                f"🔗 Linked deployment job to '{model_name}': {expected_url_no_workspace}"
            )


@pytest.mark.parametrize(
    ("version_str", "expected_is_client", "expected_major", "expected_minor"),
    [
        ("client.2.0", True, 2, 0),
        ("client.3.1", True, 3, 1),
        ("13.2", False, 13, 2),
        ("15.4", False, 15, 4),
    ],
)
def test_databricks_runtime_version_parse(
    version_str,
    expected_is_client,
    expected_major,
    expected_minor,
):
    """Test that DatabricksRuntimeVersion.parse() correctly parses version strings."""
    version = DatabricksRuntimeVersion.parse(version_str)
    assert version.is_client_image == expected_is_client
    assert version.major == expected_major
    assert version.minor == expected_minor


@pytest.mark.parametrize(
    ("env_version", "expected_is_client", "expected_major", "expected_minor"),
    [
        ("client.2.0", True, 2, 0),
        ("13.2", False, 13, 2),
    ],
)
def test_databricks_runtime_version_parse_default(
    monkeypatch,
    env_version,
    expected_is_client,
    expected_major,
    expected_minor,
):
    """Test that DatabricksRuntimeVersion.parse() works without arguments."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", env_version)
    version = DatabricksRuntimeVersion.parse()
    assert version.is_client_image == expected_is_client
    assert version.major == expected_major
    assert version.minor == expected_minor


def test_databricks_runtime_version_parse_default_no_env(monkeypatch):
    """Test that DatabricksRuntimeVersion.parse() raises error when no environment variable is
    set.
    """
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    with pytest.raises(Exception, match="Failed to parse databricks runtime version"):
        DatabricksRuntimeVersion.parse()


@pytest.mark.parametrize(
    "invalid_version",
    [
        "invalid",
        "client",
        "client.invalid",
        "13",
    ],
)
def test_databricks_runtime_version_parse_invalid(invalid_version):
    """Test that DatabricksRuntimeVersion.parse() raises error for invalid version strings."""
    with pytest.raises(Exception, match="Failed to parse databricks runtime version"):
        DatabricksRuntimeVersion.parse(invalid_version)
