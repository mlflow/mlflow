import pytest
import git
from unittest import mock

from mlflow.tracking.context.git_context import GitRunContext
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    LEGACY_MLFLOW_GIT_REPO_URL,
)

# pylint: disable=unused-argument


MOCK_SCRIPT_NAME = "/path/to/script.py"
MOCK_COMMIT_HASH = "commit-hash"
MOCK_REPO_URL = "https://mockurl.com"


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
    mock_repo_remote = mock.Mock()
    mock_repo_remote.url = MOCK_REPO_URL
    mock_repo.remotes = [mock_repo_remote]
    with mock.patch("git.Repo", return_value=mock_repo):
        yield mock_repo


def test_git_run_context_in_context_true(patch_script_name, patch_git_repo):
    assert GitRunContext().in_context()


def test_git_run_context_in_context_false(patch_script_name):
    with mock.patch("git.Repo", side_effect=git.InvalidGitRepositoryError):
        assert not GitRunContext().in_context()


def test_git_run_context_tags(patch_script_name, patch_git_repo):
    assert GitRunContext().tags() == {
        MLFLOW_GIT_COMMIT: MOCK_COMMIT_HASH,
        MLFLOW_GIT_REPO_URL: MOCK_REPO_URL,
        LEGACY_MLFLOW_GIT_REPO_URL: MOCK_REPO_URL,
    }


def test_git_run_context_caching(patch_script_name):
    """Check that the git commit hash and repo URL are only looked up once."""

    mock_repo = mock.Mock()
    mock_hexsha = mock.PropertyMock(return_value=MOCK_COMMIT_HASH)
    type(mock_repo.head.commit).hexsha = mock_hexsha
    mock_repo_remote = mock.Mock()
    mock_repo_url = mock.PropertyMock(return_value=MOCK_REPO_URL)
    type(mock_repo_remote).url = mock_repo_url
    mock_repo.remotes = [mock_repo_remote]

    with mock.patch("git.Repo", return_value=mock_repo):
        context = GitRunContext()
        context.in_context()
        context.tags()
        context.tags()

    assert mock_hexsha.call_count == 1
    assert mock_repo_url.call_count == 1
