import subprocess
from pathlib import Path
from unittest import mock

import pytest

from mlflow.genai import enable_git_model_versioning


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "test_repo"
    path.mkdir()
    subprocess.check_call(["git", "init"], cwd=path)
    subprocess.check_call(["git", "commit", "--allow-empty", "-m", "test"], cwd=path)
    monkeypatch.chdir(path)
    return path


def test_enable_git_model_versioning(monkeypatch: pytest.MonkeyPatch, tmp_git_repo: Path):
    context = enable_git_model_versioning()
    assert context.info.commit is not None
    assert context.info.branch is not None
    assert context.info.dirty is False

    # Create a dummy file to make the repo dirty
    Path(tmp_git_repo / "dummy.txt").touch()
    context = enable_git_model_versioning()
    assert context.info.dirty is True

    # Checkout a new branch
    subprocess.check_call(["git", "checkout", "-b", "new-branch"], cwd=tmp_git_repo)
    context = enable_git_model_versioning()
    assert context.info.branch == "new-branch"


def test_disable_git_model_versioning_in_non_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match=r"Git information is not available"):
        context = enable_git_model_versioning()
    assert context.info is None


def test_enable_git_model_versioning_git_unavailable(tmp_git_repo: Path):
    with mock.patch(
        "mlflow.genai.git_versioning.GitInfo._is_git_available", return_value=False
    ) as m:
        with pytest.warns(UserWarning, match=r"Git information is not available"):
            context = enable_git_model_versioning()
        assert context.info is None
        m.assert_called_once()


def test_enable_git_model_versioning_context_manager(tmp_git_repo: Path):
    with enable_git_model_versioning() as context:
        assert context.info is not None
