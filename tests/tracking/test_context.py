import mock
import pytest
import git
from six.moves import reload_module as reload

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, \
    MLFLOW_GIT_COMMIT, MLFLOW_DATABRICKS_NOTEBOOK_ID, MLFLOW_DATABRICKS_NOTEBOOK_PATH, \
    MLFLOW_DATABRICKS_WEBAPP_URL
import mlflow.tracking.context
from mlflow.tracking.context import DefaultRunContext, GitRunContext, \
    DatabricksNotebookRunContext, RunContextProviderRegistry, resolve_tags


MOCK_SCRIPT_NAME = "/path/to/script.py"
MOCK_COMMIT_HASH = "commit-hash"


@pytest.fixture
def patch_script_name():
    patch_sys_argv = mock.patch("sys.argv", [MOCK_SCRIPT_NAME])
    patch_os_path_isfile = mock.patch("os.path.isfile", return_value=False)
    with patch_sys_argv, patch_os_path_isfile:
        yield


@pytest.fixture
def patch_git_repo():
    mock_repo = mock.Mock()
    mock_repo.head.commit.hexsha = MOCK_COMMIT_HASH
    with mock.patch("git.Repo", return_value=mock_repo):
        yield mock_repo


def test_default_run_context_in_context():
    assert DefaultRunContext().in_context() is True


def test_default_run_context_tags(patch_script_name):
    mock_user = mock.Mock()
    with mock.patch("getpass.getuser", return_value=mock_user):
        assert DefaultRunContext().tags() == {
            MLFLOW_USER: mock_user,
            MLFLOW_SOURCE_NAME: MOCK_SCRIPT_NAME,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.LOCAL)
        }


def test_git_run_context_in_context_true(patch_script_name, patch_git_repo):
    assert GitRunContext().in_context()


def test_git_run_context_in_context_false(patch_script_name):
    with mock.patch("git.Repo", side_effect=git.InvalidGitRepositoryError):
        assert not GitRunContext().in_context()


def test_git_run_context_tags(patch_script_name, patch_git_repo):
    assert GitRunContext().tags() == {
        MLFLOW_GIT_COMMIT: MOCK_COMMIT_HASH
    }


def test_git_run_context_caching(patch_script_name):
    """Check that the git commit hash is only looked up once."""

    mock_repo = mock.Mock()
    mock_hexsha = mock.PropertyMock(return_value=MOCK_COMMIT_HASH)
    type(mock_repo.head.commit).hexsha = mock_hexsha

    with mock.patch("git.Repo", return_value=mock_repo):
        context = GitRunContext()
        context.in_context()
        context.tags()

    assert mock_hexsha.call_count == 1


def test_databricks_notebook_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookRunContext().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_run_context_tags():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id")
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")

    with patch_notebook_id as notebook_id_mock, patch_notebook_path as notebook_path_mock, \
            patch_webapp_url as webapp_url_mock:
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value
        }


def test_databricks_notebook_run_context_tags_nones():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id",
                                   return_value=None)
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path",
                                     return_value=None)
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url",
                                  return_value=None)

    with patch_notebook_id, patch_notebook_path, patch_webapp_url:
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }


def test_run_context_provider_registry_register():
    provider_class = mock.Mock()

    registry = RunContextProviderRegistry()
    registry.register(provider_class)

    assert set(registry) == {provider_class.return_value}


def test_run_context_provider_registry_register_entrypoints():
    provider_class = mock.Mock()
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = provider_class

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RunContextProviderRegistry()
        registry.register_entrypoints()

    assert set(registry) == {provider_class.return_value}
    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


@pytest.mark.parametrize("exception",
                         [AttributeError("test exception"), ImportError("test exception")])
def test_run_context_provider_registry_register_entrypoints_handles_exception(exception):
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.side_effect = exception

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RunContextProviderRegistry()
        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


def _currently_registered_run_context_provider_classes():
    return {
        provider.__class__
        for provider in mlflow.tracking.context._run_context_provider_registry
    }


def test_registry_instance_defaults():
    expected_classes = {DefaultRunContext, GitRunContext, DatabricksNotebookRunContext}
    assert expected_classes.issubset(_currently_registered_run_context_provider_classes())


def test_registry_instance_loads_entrypoints():

    class MockRunContext(object):
        pass

    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = MockRunContext

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        # Entrypoints are registered at import time, so we need to reload the module to register th
        # entrypoint given by the mocked extrypoints.get_group_all
        reload(mlflow.tracking.context)

    assert MockRunContext in _currently_registered_run_context_provider_classes()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


@pytest.mark.large
def test_run_context_provider_registry_with_installed_plugin(tmp_wkdir):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.context)

    from mlflow_test_plugin import PluginRunContextProvider
    assert PluginRunContextProvider in _currently_registered_run_context_provider_classes()

    # The test plugin's context provider always returns False from in_context
    # to avoid polluting tags in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(PluginRunContextProvider, "in_context", return_value=True):
        assert resolve_tags()["test"] == "tag"


@pytest.fixture
def mock_run_context_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.tags.return_value = {"one": "one-val", "two": "two-val", "three": "three-val"}

    skipped_provider = mock.Mock()
    skipped_provider.in_context.return_value = False

    override_provider = mock.Mock()
    override_provider.in_context.return_value = True
    override_provider.tags.return_value = {"one": "override", "new": "new-val"}

    providers = [base_provider, skipped_provider, override_provider]

    with mock.patch("mlflow.tracking.context._run_context_provider_registry", providers):
        yield

    skipped_provider.tags.assert_not_called()


def test_resolve_tags(mock_run_context_providers):
    tags_arg = {"two": "arg-override", "arg": "arg-val"}
    assert resolve_tags(tags_arg) == {
        "one": "override",
        "two": "arg-override",
        "three": "three-val",
        "new": "new-val",
        "arg": "arg-val"
    }


def test_resolve_tags_no_arg(mock_run_context_providers):
    assert resolve_tags() == {
        "one": "override",
        "two": "two-val",
        "three": "three-val",
        "new": "new-val"
    }
