"""
Integration tests for UV package manager support in model logging and loading.

Tests the end-to-end workflow:
1. UV project detection during log_model()
2. Artifact generation (uv.lock, pyproject.toml, .python-version, requirements.txt)
3. Model loading with UV artifacts

These tests use REAL UV calls (not mocked) where possible, following MLflow best practices.
Tests requiring UV are skipped if UV is not installed or below minimum version.
"""

import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

import mlflow
import mlflow.pyfunc
from mlflow.utils.uv_utils import (
    _PYPROJECT_FILE,
    _PYTHON_VERSION_FILE,
    _UV_LOCK_FILE,
    is_uv_available,
)

# Constants for artifact file names
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"

# Skip marker for tests requiring UV
requires_uv = pytest.mark.skipif(
    not is_uv_available(),
    reason="UV is not installed or below minimum required version (0.5.0)",
)


class SimplePythonModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture
def python_model():
    return SimplePythonModel()


@pytest.fixture
def uv_project_real(tmp_path):
    """Create a real UV project with uv lock."""
    # Create pyproject.toml
    pyproject_content = """[project]
name = "test_uv_project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)

    # Create .python-version
    (tmp_path / _PYTHON_VERSION_FILE).write_text("3.11.5\n")

    # Create minimal package structure for hatchling
    pkg_dir = tmp_path / "test_uv_project"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Test UV project."""\n__version__ = "0.1.0"\n')

    # Run uv lock to generate real uv.lock
    result = subprocess.run(
        ["uv", "lock"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"uv lock failed: {result.stderr}")

    return tmp_path


@pytest.fixture
def uv_project_real_no_python_version(tmp_path):
    """Create a real UV project without .python-version file."""
    pyproject_content = """[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)

    result = subprocess.run(
        ["uv", "lock"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"uv lock failed: {result.stderr}")

    return tmp_path


# --- Model Logging Tests with Real UV ---


@requires_uv
def test_pyfunc_log_model_copies_uv_artifacts(uv_project_real, python_model, monkeypatch):
    monkeypatch.chdir(uv_project_real)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify UV artifacts are copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert (artifact_dir / _PYTHON_VERSION_FILE).exists()

        # Verify content matches source
        assert "version = 1" in (artifact_dir / _UV_LOCK_FILE).read_text()
        assert "test_uv_project" in (artifact_dir / _PYPROJECT_FILE).read_text()
        assert "3.11.5" in (artifact_dir / _PYTHON_VERSION_FILE).read_text()


@requires_uv
def test_pyfunc_log_model_uses_uv_python_version_in_python_env(
    uv_project_real, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project_real)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify python_env.yaml uses UV project's Python version
        python_env_file = artifact_dir / _PYTHON_ENV_FILE_NAME
        assert python_env_file.exists()
        python_env_content = python_env_file.read_text()
        assert "3.11.5" in python_env_content


@requires_uv
def test_pyfunc_log_model_extracts_python_version_from_pyproject(
    uv_project_real_no_python_version, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project_real_no_python_version)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify python_env.yaml extracts version from pyproject.toml
        python_env_file = artifact_dir / _PYTHON_ENV_FILE_NAME
        assert python_env_file.exists()
        python_env_content = python_env_file.read_text()
        # Should extract "3.11" from requires-python = ">=3.11"
        assert "3.11" in python_env_content


@requires_uv
def test_pyfunc_log_model_respects_mlflow_log_uv_files_env_var(
    uv_project_real, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project_real)
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", "false")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # UV artifacts should NOT be copied when env var is false
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()

        # But requirements.txt should still exist (from UV export)
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()
        requirements_content = (artifact_dir / _REQUIREMENTS_FILE_NAME).read_text()
        assert "numpy" in requirements_content.lower()


@requires_uv
def test_pyfunc_log_model_with_explicit_uv_project_path_parameter(
    tmp_path, uv_project_real, python_model, monkeypatch
):
    # Work from a different directory than the UV project
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with mlflow.start_run() as run:
        # Use explicit uv_project_path parameter to point to UV project
        mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            uv_project_path=uv_project_real,
        )

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify UV artifacts from explicit path are copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert "test_uv_project" in (artifact_dir / _PYPROJECT_FILE).read_text()


@requires_uv
def test_pyfunc_log_model_generates_requirements_from_uv_export(
    uv_project_real, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project_real)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify requirements.txt exists and contains numpy from UV export
        requirements_file = artifact_dir / _REQUIREMENTS_FILE_NAME
        assert requirements_file.exists()
        requirements_content = requirements_file.read_text()
        assert "numpy" in requirements_content.lower()


# --- Fallback Tests (mocking required to simulate UV unavailable) ---


def test_pyfunc_log_model_falls_back_when_uv_not_available(tmp_path, python_model, monkeypatch):
    # Create a UV project structure (files exist but UV unavailable)
    (tmp_path / _UV_LOCK_FILE).write_text('version = 1\nrequires-python = ">=3.10"\n')
    (tmp_path / _PYPROJECT_FILE).write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=False),
        mlflow.start_run() as run,
    ):
        # Should not raise, falls back to pip inference
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # requirements.txt should still exist (from pip inference)
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


def test_pyfunc_log_model_falls_back_when_uv_export_fails(tmp_path, python_model, monkeypatch):
    # Create a UV project structure
    (tmp_path / _UV_LOCK_FILE).write_text('version = 1\nrequires-python = ">=3.10"\n')
    (tmp_path / _PYPROJECT_FILE).write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")),
        mlflow.start_run() as run,
    ):
        # Should not raise, falls back to pip inference
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # requirements.txt should still exist (from pip inference fallback)
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


def test_pyfunc_log_model_non_uv_project_uses_standard_inference(
    python_model, tmp_path, monkeypatch
):
    # Empty directory - not a UV project
    monkeypatch.chdir(tmp_path)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Standard artifacts should exist
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()
        assert (artifact_dir / _PYTHON_ENV_FILE_NAME).exists()

        # UV artifacts should NOT exist
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()


# --- Model Loading Tests ---


@requires_uv
def test_load_pyfunc_model_with_uv_artifacts(uv_project_real, python_model, monkeypatch):
    monkeypatch.chdir(uv_project_real)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)
        model_uri = f"runs:/{run.info.run_id}/model"

        # Load as pyfunc
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        assert loaded_model is not None
        assert loaded_model.metadata is not None


@requires_uv
def test_load_pyfunc_model_can_predict_after_loading(uv_project_real, python_model, monkeypatch):
    monkeypatch.chdir(uv_project_real)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)
        model_uri = f"runs:/{run.info.run_id}/model"

        # Load and predict
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        import pandas as pd

        test_input = pd.DataFrame({"a": [1, 2, 3]})
        predictions = loaded_model.predict(test_input)

        # SimplePythonModel returns input as-is
        assert predictions is not None


# --- Save Model Tests ---


@requires_uv
def test_pyfunc_save_model_with_uv_project(uv_project_real, python_model, tmp_path, monkeypatch):
    monkeypatch.chdir(uv_project_real)
    model_path = tmp_path / "saved_model"

    mlflow.pyfunc.save_model(model_path, python_model=python_model)

    # Verify artifacts
    assert (model_path / _REQUIREMENTS_FILE_NAME).exists()
    assert (model_path / _UV_LOCK_FILE).exists()
    assert (model_path / _PYPROJECT_FILE).exists()
    assert (model_path / _PYTHON_VERSION_FILE).exists()


@requires_uv
def test_pyfunc_save_model_with_explicit_uv_project_path(
    uv_project_real, python_model, tmp_path, monkeypatch
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    model_path = tmp_path / "saved_model"
    monkeypatch.chdir(work_dir)

    mlflow.pyfunc.save_model(
        model_path,
        python_model=python_model,
        uv_project_path=uv_project_real,
    )

    # Verify UV artifacts from explicit path
    assert (model_path / _UV_LOCK_FILE).exists()
    assert (model_path / _PYPROJECT_FILE).exists()


# --- Environment Variable Variations ---


@requires_uv
@pytest.mark.parametrize("env_value", ["false", "0", "no", "FALSE", "No"])
def test_mlflow_log_uv_files_env_var_false_variants(
    uv_project_real, python_model, monkeypatch, env_value
):
    monkeypatch.chdir(uv_project_real)
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # UV artifacts should NOT be copied
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()

        # requirements.txt should still exist
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


@requires_uv
@pytest.mark.parametrize("env_value", ["true", "1", "yes", "TRUE"])
def test_mlflow_log_uv_files_env_var_true_variants(
    uv_project_real, python_model, monkeypatch, env_value
):
    monkeypatch.chdir(uv_project_real)
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # UV artifacts should be copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()


# --- Phase 2: Dependency Groups Integration Tests ---


@pytest.fixture
def uv_project_with_groups(tmp_path):
    """Create a UV project with dependency groups."""
    pyproject_content = """[project]
name = "test_uv_groups"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
]

[project.optional-dependencies]
gpu = ["scipy>=1.10.0"]

[dependency-groups]
serving = ["gunicorn>=21.0.0"]
dev = ["pytest>=7.0.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)
    (tmp_path / _PYTHON_VERSION_FILE).write_text("3.11.5\n")

    # Create minimal package structure for hatchling
    pkg_dir = tmp_path / "test_uv_groups"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Test UV groups project."""\n__version__ = "0.1.0"\n')

    result = subprocess.run(
        ["uv", "lock"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"uv lock failed: {result.stderr}")

    return tmp_path


@requires_uv
def test_export_uv_requirements_with_groups_real(uv_project_with_groups):
    from mlflow.utils.uv_utils import export_uv_requirements

    result = export_uv_requirements(uv_project_with_groups, groups=["serving"])

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    # Should have numpy (project dep) and gunicorn (serving group)
    assert "numpy" in pkg_names
    assert "gunicorn" in pkg_names
    # Should NOT have pytest (dev group not included)
    assert "pytest" not in pkg_names


@requires_uv
def test_export_uv_requirements_with_only_groups_real(uv_project_with_groups):
    from mlflow.utils.uv_utils import export_uv_requirements

    result = export_uv_requirements(uv_project_with_groups, only_groups=["serving"])

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    # Should have ONLY gunicorn (serving group), NOT numpy (project dep)
    assert "gunicorn" in pkg_names
    # numpy may or may not be included depending on UV version behavior


@requires_uv
def test_export_uv_requirements_with_extras_real(uv_project_with_groups):
    from mlflow.utils.uv_utils import export_uv_requirements

    result = export_uv_requirements(uv_project_with_groups, extras=["gpu"])

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    # Should have numpy (project dep) and scipy (gpu extra)
    assert "numpy" in pkg_names
    assert "scipy" in pkg_names


@requires_uv
def test_export_uv_requirements_with_env_vars(uv_project_with_groups, monkeypatch):
    from mlflow.utils.uv_utils import (
        export_uv_requirements,
        get_uv_extras_from_env,
        get_uv_groups_from_env,
    )

    monkeypatch.setenv("MLFLOW_UV_GROUPS", "serving")
    monkeypatch.setenv("MLFLOW_UV_EXTRAS", "gpu")

    groups = get_uv_groups_from_env()
    extras = get_uv_extras_from_env()

    assert groups == ["serving"]
    assert extras == ["gpu"]

    result = export_uv_requirements(uv_project_with_groups, groups=groups, extras=extras)

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    assert "gunicorn" in pkg_names
    assert "scipy" in pkg_names


# --- Phase 2: UV Sync Environment Setup Integration Tests ---


@requires_uv
def test_setup_uv_sync_environment_real(uv_project_real, tmp_path):
    from mlflow.utils.uv_utils import has_uv_lock_artifact, setup_uv_sync_environment

    # Simulate model artifacts by copying UV files
    model_artifacts = tmp_path / "model_artifacts"
    model_artifacts.mkdir()

    shutil.copy2(uv_project_real / _UV_LOCK_FILE, model_artifacts / _UV_LOCK_FILE)
    shutil.copy2(uv_project_real / _PYTHON_VERSION_FILE, model_artifacts / _PYTHON_VERSION_FILE)

    assert has_uv_lock_artifact(model_artifacts)

    # Setup sync environment
    env_dir = tmp_path / "env"
    result = setup_uv_sync_environment(env_dir, model_artifacts, "3.11.5")

    assert result is True
    assert (env_dir / _UV_LOCK_FILE).exists()
    assert (env_dir / _PYPROJECT_FILE).exists()
    assert (env_dir / _PYTHON_VERSION_FILE).exists()

    # Verify pyproject.toml content
    pyproject_content = (env_dir / _PYPROJECT_FILE).read_text()
    assert 'name = "mlflow-model-env"' in pyproject_content
    assert 'requires-python = ">=3.11"' in pyproject_content


@requires_uv
def test_extract_index_urls_from_real_uv_lock(uv_project_real):
    from mlflow.utils.uv_utils import extract_index_urls_from_uv_lock

    # Real uv.lock for a simple numpy project should only have PyPI sources
    # (and possibly other well-known public indexes like PyTorch CPU wheels).
    # The key guarantee: no truly private indexes should appear.
    result = extract_index_urls_from_uv_lock(uv_project_real / _UV_LOCK_FILE)

    # Allow well-known public indexes that may appear in CI environments
    well_known_public = {"https://download.pytorch.org/whl/cpu"}
    truly_private = [url for url in result if url not in well_known_public]
    assert truly_private == []


@requires_uv
def test_run_uv_sync_real(uv_project_real, tmp_path):
    from mlflow.utils.uv_utils import run_uv_sync

    # Copy the entire UV project to a new directory for sync
    # Using original pyproject.toml and uv.lock ensures lockfile matches
    sync_dir = tmp_path / "sync_project"
    shutil.copytree(uv_project_real, sync_dir)

    # Run uv sync in the copied project
    result = run_uv_sync(sync_dir, frozen=True, no_dev=True)

    assert result is True
    # Verify .venv was created
    assert (sync_dir / ".venv").exists()
