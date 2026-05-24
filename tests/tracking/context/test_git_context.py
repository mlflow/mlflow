from unittest import mock

import git
import pytest

from mlflow.tracking.context.git_context import GitRunContext
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_DIRTY,
    MLFLOW_GIT_REPO_URL,
)

MOCK_SCRIPT_NAME = "/path/to/script.py"
MOCK_COMMIT_HASH = "commit-hash"
MOCK_BRANCH_NAME = "main"
MOCK_REPO_URL = "https://github.com/user/repo.git"


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
    mock_repo.ignored.return_value = []
    mock_repo.active_branch.name = MOCK_BRANCH_NAME
    mock_repo.is_dirty.return_value = False
    mock_remote = mock.Mock()
    mock_remote.url = MOCK_REPO_URL
    mock_repo.remotes = [mock_remote]
    with mock.patch("git.Repo", return_value=mock_repo):
        yield mock_repo


def test_git_run_context_in_context_true(patch_script_name, patch_git_repo):
    assert GitRunContext().in_context()


def test_git_run_context_in_context_false(patch_script_name):
    with mock.patch("git.Repo", side_effect=git.InvalidGitRepositoryError):
        assert not GitRunContext().in_context()


def test_git_run_context_tags(patch_script_name, patch_git_repo):
    tags = GitRunContext().tags()
    assert tags[MLFLOW_GIT_COMMIT] == MOCK_COMMIT_HASH
    assert tags[MLFLOW_GIT_BRANCH] == MOCK_BRANCH_NAME
    assert tags[MLFLOW_GIT_REPO_URL] == MOCK_REPO_URL
    assert tags[MLFLOW_GIT_DIRTY] == "false"


def test_git_run_context_tags_dirty_repo(patch_script_name):
    mock_repo = mock.Mock()
    mock_repo.head.commit.hexsha = MOCK_COMMIT_HASH
    mock_repo.ignored.return_value = []
    mock_repo.active_branch.name = MOCK_BRANCH_NAME
    mock_repo.is_dirty.return_value = True
    mock_remote = mock.Mock()
    mock_remote.url = MOCK_REPO_URL
    mock_repo.remotes = [mock_remote]
    with mock.patch("git.Repo", return_value=mock_repo):
        tags = GitRunContext().tags()
    assert tags[MLFLOW_GIT_DIRTY] == "true"


def test_git_run_context_tags_no_remotes(patch_script_name):
    mock_repo = mock.Mock()
    mock_repo.head.commit.hexsha = MOCK_COMMIT_HASH
    mock_repo.ignored.return_value = []
    mock_repo.active_branch.name = MOCK_BRANCH_NAME
    mock_repo.is_dirty.return_value = False
    mock_repo.remotes = []
    with mock.patch("git.Repo", return_value=mock_repo):
        tags = GitRunContext().tags()
    assert tags[MLFLOW_GIT_COMMIT] == MOCK_COMMIT_HASH
    assert tags[MLFLOW_GIT_BRANCH] == MOCK_BRANCH_NAME
    assert MLFLOW_GIT_REPO_URL not in tags


def test_git_run_context_tags_detached_head(patch_script_name):
    mock_repo = mock.Mock()
    mock_repo.head.commit.hexsha = MOCK_COMMIT_HASH
    mock_repo.ignored.return_value = []
    mock_repo.active_branch = mock.PropertyMock(side_effect=TypeError("HEAD is detached"))
    type(mock_repo).active_branch = mock.PropertyMock(side_effect=TypeError("HEAD is detached"))
    mock_repo.is_dirty.return_value = False
    mock_repo.remotes = []
    with mock.patch("git.Repo", return_value=mock_repo):
        tags = GitRunContext().tags()
    assert tags[MLFLOW_GIT_COMMIT] == MOCK_COMMIT_HASH
    assert MLFLOW_GIT_BRANCH not in tags


def test_git_run_context_caching(patch_script_name):
    """Verify that _resolve() creates git.Repo exactly once, regardless of how
    many times in_context() and tags() are called."""
    with mock.patch("git.Repo") as mock_repo:
        context = GitRunContext()
        context.in_context()
        context.tags()

        # The single _resolve call should have instantiated Repo exactly once.
        assert mock_repo.call_count == 1

        # Calling again should NOT create additional Repo instances.
        context.in_context()
        context.tags()
        context.in_context()
        context.tags()

        mock_repo.assert_called_once()
