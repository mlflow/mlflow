import mock
import pytest
import git

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, MLFLOW_GIT_COMMIT, \
    MLFLOW_DATABRICKS_NOTEBOOK_ID, MLFLOW_DATABRICKS_NOTEBOOK_PATH, MLFLOW_DATABRICKS_WEBAPP_URL
from mlflow.tracking.context import DefaultContext, GitContext, DatabricksNotebookContext, \
    resolve_tags


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


def test_default_context_in_context():
    assert DefaultContext().in_context() is True


def test_default_context_tags(patch_script_name):
    assert DefaultContext().tags() == {
        MLFLOW_SOURCE_NAME: MOCK_SCRIPT_NAME,
        MLFLOW_SOURCE_TYPE: SourceType.LOCAL
    }


def test_git_context_in_context_true(patch_script_name, patch_git_repo):
    assert GitContext().in_context()


def test_git_context_in_context_false(patch_script_name):
    with mock.patch("git.Repo", side_effect=git.InvalidGitRepositoryError):
        assert not GitContext().in_context()


def test_git_context_tags(patch_script_name, patch_git_repo):
    assert GitContext().tags() == {
        MLFLOW_GIT_COMMIT: MOCK_COMMIT_HASH
    }


def test_databricks_notebook_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookContext().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_tags():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id")
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")

    with patch_notebook_id as notebook_id_mock, patch_notebook_path as notebook_path_mock, \
            patch_webapp_url as webapp_url_mock:
        assert DatabricksNotebookContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.NOTEBOOK,
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value
        }


def test_databricks_notebook_tags_nones():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id",
                                   return_value=None)
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path",
                                     return_value=None)
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url",
                                  return_value=None)

    with patch_notebook_id, patch_notebook_path, patch_webapp_url:
        assert DatabricksNotebookContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.NOTEBOOK,
        }


@pytest.fixture
def mock_context_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.tags.return_value = {"one": "one-val", "two": "two-val", "three": "three-val"}

    skipped_provider = mock.Mock()
    skipped_provider.in_context.return_value = False

    override_provider = mock.Mock()
    override_provider.in_context.return_value = True
    override_provider.tags.return_value = {"one": "override", "new": "new-val"}

    providers = [base_provider, skipped_provider, override_provider]

    with mock.patch("mlflow.tracking.context._context_provider_registry", providers):
        yield

    skipped_provider.tags.assert_not_called()


def test_resolve_tags(mock_context_providers):
    tags_arg = {"two": "arg-override", "arg": "arg-val"}
    assert resolve_tags(tags_arg) == {
        "one": "override",
        "two": "arg-override",
        "three": "three-val",
        "new": "new-val",
        "arg": "arg-val"
    }


def test_resolve_tags_no_arg(mock_context_providers):
    assert resolve_tags() == {
        "one": "override",
        "two": "two-val",
        "three": "three-val",
        "new": "new-val"
    }
