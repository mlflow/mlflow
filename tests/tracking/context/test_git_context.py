from unittest import mock

import git
import pytest

from mlflow.tracking.context.git_context import GitRunContext
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

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
    mock_repo.ignored.return_value = []
    with mock.patch("git.Repo", return_value=mock_repo):
        yield mock_repo


def test_git_run_context_in_context_true(patch_script_name, patch_git_repo):
    assert GitRunContext().in_context()


def test_git_run_context_in_context_false(patch_script_name):
    with mock.patch("git.Repo", side_effect=git.InvalidGitRepositoryError):
        assert not GitRunContext().in_context()


def test_git_run_context_tags(patch_script_name, patch_git_repo):
    assert GitRunContext().tags() == {MLFLOW_GIT_COMMIT: MOCK_COMMIT_HASH}


def test_git_run_context_caching(patch_script_name):
    """Check that the git commit hash is only looked up once."""

    with mock.patch("git.Repo") as mock_repo:
        context = GitRunContext()
        context.in_context()
        context.tags()

    mock_repo.assert_called_once()
