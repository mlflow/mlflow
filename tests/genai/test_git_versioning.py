import subprocess
from pathlib import Path

import pytest

import mlflow
from mlflow.genai import disable_git_model_versioning, enable_git_model_versioning
from mlflow.genai.git_versioning import _get_active_git_context


@pytest.fixture(autouse=True)
def cleanup_active_context():
    yield
    disable_git_model_versioning()


TEST_FILENAME = "test.txt"


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "test_repo"
    path.mkdir()
    subprocess.check_call(["git", "init"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=path)
    (path / TEST_FILENAME).touch()
    subprocess.check_call(["git", "add", "."], cwd=path)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=path)
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
    assert context.active_model is None


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


def test_enable_git_model_versioning_creates_initial_logged_model(tmp_git_repo: Path):
    with enable_git_model_versioning() as context:
        assert mlflow.get_active_model_id() == context.active_model.model_id
        models = mlflow.search_logged_models(output_format="list")
        assert len(models) == 1
        assert models[0].model_id == context.active_model.model_id
        assert models[0].tags.items() >= context.info.to_mlflow_tags().items()
    assert mlflow.get_active_model_id() is None


def test_enable_git_model_versioning_reuses_model_when_no_changes(tmp_git_repo: Path):
    # Create initial model
    with enable_git_model_versioning() as context:
        initial_model_id = context.active_model.model_id
    assert mlflow.get_active_model_id() is None

    # No git state changes, should reuse the same model
    with enable_git_model_versioning() as context:
        assert mlflow.get_active_model_id() == initial_model_id
        models = mlflow.search_logged_models(output_format="list")
        assert len(models) == 1
        assert models[0].model_id == initial_model_id
    assert mlflow.get_active_model_id() is None


def test_enable_git_model_versioning_creates_new_model_on_commit(tmp_git_repo: Path):
    # Create initial model
    with enable_git_model_versioning() as context:
        initial_model_id = context.active_model.model_id
    assert mlflow.get_active_model_id() is None

    # Make a new commit
    subprocess.check_call(["git", "commit", "--allow-empty", "-m", "commit"], cwd=tmp_git_repo)

    # Should create a new logged model
    with enable_git_model_versioning() as context:
        assert mlflow.get_active_model_id() != initial_model_id
        assert mlflow.get_active_model_id() == context.active_model.model_id
        models = mlflow.search_logged_models(output_format="list")
        assert len(models) == 2
        assert models[0].model_id == context.active_model.model_id
        assert models[0].tags.items() >= context.info.to_mlflow_tags().items()
    assert mlflow.get_active_model_id() is None


def test_enable_git_model_versioning_creates_new_model_on_dirty_repo(tmp_git_repo: Path):
    # Create initial model
    with enable_git_model_versioning() as context:
        initial_model_id = context.active_model.model_id
    assert mlflow.get_active_model_id() is None

    # Modify a tracked file to make the repo dirty
    (tmp_git_repo / TEST_FILENAME).write_text("Updated content")

    # Should create a new logged model
    with enable_git_model_versioning() as context:
        assert mlflow.get_active_model_id() != initial_model_id
        assert mlflow.get_active_model_id() == context.active_model.model_id
        models = mlflow.search_logged_models(output_format="list")
        assert len(models) == 2
        assert models[0].model_id == context.active_model.model_id
        assert models[0].tags.items() >= context.info.to_mlflow_tags().items()
    assert mlflow.get_active_model_id() is None


def test_enable_git_model_versioning_ignores_untracked_files(tmp_git_repo: Path):
    # Create initial model
    with enable_git_model_versioning() as context:
        initial_model_id = context.active_model.model_id
    assert mlflow.get_active_model_id() is None

    # Create an untracked file
    (tmp_git_repo / "untracked.txt").touch()

    # Should NOT create a new logged model
    with enable_git_model_versioning() as context:
        assert mlflow.get_active_model_id() == initial_model_id
        models = mlflow.search_logged_models(output_format="list")
        assert len(models) == 1
        assert models[0].model_id == initial_model_id
    assert mlflow.get_active_model_id() is None


def test_enable_git_model_versioning_default_remote_name(tmp_git_repo: Path):
    subprocess.check_call(
        ["git", "remote", "add", "origin", "https://github.com/test/repo.git"], cwd=tmp_git_repo
    )
    context = enable_git_model_versioning()
    assert context.info.repo_url == "https://github.com/test/repo.git"


def test_enable_git_model_versioning_custom_remote_name(tmp_git_repo: Path):
    # Add multiple remotes
    subprocess.check_call(
        ["git", "remote", "add", "origin", "https://github.com/test/repo.git"],
        cwd=tmp_git_repo,
    )
    subprocess.check_call(
        ["git", "remote", "add", "upstream", "https://github.com/upstream/repo.git"],
        cwd=tmp_git_repo,
    )
    context = enable_git_model_versioning(remote_name="upstream")
    assert context.info.repo_url == "https://github.com/upstream/repo.git"


def test_enable_git_model_versioning_no_remote(tmp_git_repo: Path):
    # No remote - repo_url should be None
    context = enable_git_model_versioning()
    assert context.info.repo_url is None
