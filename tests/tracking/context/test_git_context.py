from unittest import mock

import git
import pytest

from mlflow.tracking.context.git_context import GitRunContext
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH, MLFLOW_GIT_REPO_URL

MOCK_SCRIPT_NAME = "/path/to/script.py"
MOCK_COMMIT_HASH = "commit-hash"
MOCK_BRANCH_NAME = "main"
MOCK_REPO_URL = "https://github.com/mlflow/mlflow.git"


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
    # Mock branch
    mock_repo.active_branch.name = MOCK_BRANCH_NAME
    # Mock remotes for repo URL
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
    expected_tags = {
        MLFLOW_GIT_COMMIT: MOCK_COMMIT_HASH,
        MLFLOW_GIT_BRANCH: MOCK_BRANCH_NAME,
        MLFLOW_GIT_REPO_URL: MOCK_REPO_URL
    }
    assert GitRunContext().tags() == expected_tags


def test_git_run_context_caching(patch_script_name):
    """Check that git operations are cached properly."""

    with mock.patch("mlflow.tracking.context.git_context._get_source_version") as mock_commit, \
         mock.patch("mlflow.tracking.context.git_context._get_git_branch") as mock_branch, \
         mock.patch("mlflow.tracking.context.git_context._get_git_repo_url") as mock_repo_url:
        
        mock_commit.return_value = MOCK_COMMIT_HASH
        mock_branch.return_value = MOCK_BRANCH_NAME
        mock_repo_url.return_value = MOCK_REPO_URL
        
        context = GitRunContext()
        
        # First calls should invoke the git functions
        assert context._source_version == MOCK_COMMIT_HASH
        assert context._git_branch == MOCK_BRANCH_NAME
        assert context._git_repo_url == MOCK_REPO_URL
        
        # Second calls should use cached values
        assert context._source_version == MOCK_COMMIT_HASH
        assert context._git_branch == MOCK_BRANCH_NAME
        assert context._git_repo_url == MOCK_REPO_URL
        
        # Each function should only be called once (due to caching)
        mock_commit.assert_called_once()
        mock_branch.assert_called_once()
        mock_repo_url.assert_called_once()
