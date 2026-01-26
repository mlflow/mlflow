"""
Integration tests for UV package manager support in model logging and loading.

Tests the end-to-end workflow:
1. UV project detection during log_model()
2. Artifact generation (uv.lock, pyproject.toml, .python-version, requirements.txt)
3. Model loading with UV artifacts

These tests use REAL UV calls (not mocked) where possible, following MLflow best practices.
Tests requiring UV are skipped if UV is not installed or below minimum version.
"""

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
name = "test-uv-project"
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
        assert "test-uv-project" in (artifact_dir / _PYPROJECT_FILE).read_text()
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
def test_pyfunc_log_model_with_explicit_uv_lock_parameter(
    tmp_path, uv_project_real, python_model, monkeypatch
):
    # Work from a different directory than the UV project
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with mlflow.start_run() as run:
        # Use explicit uv_lock parameter to point to UV project
        mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            uv_lock=uv_project_real / _UV_LOCK_FILE,
        )

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify UV artifacts from explicit path are copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert "test-uv-project" in (artifact_dir / _PYPROJECT_FILE).read_text()


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
def test_pyfunc_save_model_with_explicit_uv_lock(
    uv_project_real, python_model, tmp_path, monkeypatch
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    model_path = tmp_path / "saved_model"
    monkeypatch.chdir(work_dir)

    mlflow.pyfunc.save_model(
        model_path,
        python_model=python_model,
        uv_lock=uv_project_real / _UV_LOCK_FILE,
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
