import subprocess
from pathlib import Path

import pytest

from mlflow.genai import disable_git_model_versioning, enable_git_model_versioning
from mlflow.genai.git_versioning import _get_active_git_context


@pytest.fixture(autouse=True)
def cleanup_active_context():
    yield
    disable_git_model_versioning()


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "test_repo"
    path.mkdir()
    subprocess.check_call(["git", "init"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=path)
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
    # Untracked files should not be considered dirty
    assert context.info.dirty is False

    # Checkout a new branch
    subprocess.check_call(["git", "checkout", "-b", "new-branch"], cwd=tmp_git_repo)
    context = enable_git_model_versioning()
    assert context.info.branch == "new-branch"


def test_disable_git_model_versioning_in_non_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match=r"Encountered an error while retrieving git information"):
        context = enable_git_model_versioning()
    assert context.info is None


def test_enable_git_model_versioning_context_manager(tmp_git_repo: Path):
    assert _get_active_git_context() is None

    with enable_git_model_versioning() as context:
        assert _get_active_git_context() is context

    assert _get_active_git_context() is None


def test_disable_git_model_versioning_resets_context(tmp_git_repo: Path):
    with enable_git_model_versioning() as context:
        assert _get_active_git_context() is context
        disable_git_model_versioning()
        assert _get_active_git_context() is None


def test_enable_git_model_versioning_sets_active_context(tmp_git_repo: Path):
    assert _get_active_git_context() is None

    context = enable_git_model_versioning()
    assert _get_active_git_context() is context

    disable_git_model_versioning()
    assert _get_active_git_context() is None
