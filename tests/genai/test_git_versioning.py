import subprocess
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.genai import disable_git_model_versioning, enable_git_model_versioning
from mlflow.genai.git_versioning import _get_active_git_context
from mlflow.utils.mlflow_tags import MLFLOW_GIT_DIFF


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
    assert context.info.diff is None  # Clean repo has no diff

    # Create a dummy file to make the repo dirty
    Path(tmp_git_repo / "dummy.txt").touch()
    context = enable_git_model_versioning()
    # Untracked files should not be considered dirty
    assert context.info.dirty is False
    assert context.info.diff is None  # No diff for untracked files

    # Checkout a new branch
    subprocess.check_call(["git", "checkout", "-b", "new-branch"], cwd=tmp_git_repo)
    context = enable_git_model_versioning()
    assert context.info.branch == "new-branch"


def test_disable_git_model_versioning_in_non_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    with mock.patch("mlflow.genai.git_versioning._logger.warning") as mock_warning:
        context = enable_git_model_versioning()

    mock_warning.assert_called_once()
    warning_message = mock_warning.call_args[0][0]
    assert "Encountered an error while retrieving git information" in warning_message
    assert "Git model versioning is disabled" in warning_message
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


def test_git_diff_collected_when_dirty(tmp_git_repo: Path):
    # Initially clean repo
    context = enable_git_model_versioning()
    assert context.info.dirty is False
    assert context.info.diff is None
    disable_git_model_versioning()

    # Modify a tracked file
    test_file = tmp_git_repo / TEST_FILENAME
    test_file.write_text("Modified content")

    # Should have diff now
    context = enable_git_model_versioning()
    assert context.info.dirty is True
    assert context.info.diff is not None
    assert "Modified content" in context.info.diff
    assert MLFLOW_GIT_DIFF in context.info.to_mlflow_tags()

    # Make another change
    with open(test_file, "a") as f:
        f.write("\nAnother change")

    # Both changes should be in the diff
    context = enable_git_model_versioning()
    model = mlflow.get_logged_model(context.active_model.model_id)
    assert "Modified content" in model.tags[MLFLOW_GIT_DIFF]
    assert "Another change" in model.tags[MLFLOW_GIT_DIFF]


def test_git_diff_includes_staged_changes(tmp_git_repo: Path):
    # Create two files
    file1 = tmp_git_repo / "file1.txt"
    file2 = tmp_git_repo / "file2.txt"
    file1.write_text("file1 content")
    file2.write_text("file2 content")

    # Stage file1
    subprocess.check_call(["git", "add", "file1.txt"], cwd=tmp_git_repo)

    # file2 remains unstaged (but untracked files don't show in diff)
    # So let's modify an existing tracked file instead
    (tmp_git_repo / TEST_FILENAME).write_text("modified content")

    context = enable_git_model_versioning()
    assert context.info.dirty is True
    assert context.info.diff is not None
    assert "file1 content" in context.info.diff  # Staged changes
    assert "modified content" in context.info.diff  # Unstaged changes


def test_enable_git_model_versioning_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch, tmp_git_repo: Path
):
    subdir = tmp_git_repo / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    context = enable_git_model_versioning()
    assert context.info is not None
    assert context.info.commit is not None
    assert context.info.branch is not None
    assert context.info.dirty is False
