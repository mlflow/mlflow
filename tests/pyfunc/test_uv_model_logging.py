"""
Integration tests for UV package manager support in model logging and loading.

Tests the end-to-end workflow:
1. UV project detection during log_model()
2. Artifact generation (uv.lock, pyproject.toml, .python-version, requirements.txt)
3. Model loading with UV artifacts

Note: UV file copying (uv.lock, pyproject.toml, .python-version) is implemented in
mlflow.pyfunc.log_model/save_model. The sklearn flavor uses the standard requirements
inference via UV export but does not copy UV artifacts directly.
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
)

# Constants for artifact file names
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"


class SimplePythonModel(mlflow.pyfunc.PythonModel):
    """Simple PythonModel for testing."""

    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture
def python_model():
    return SimplePythonModel()


@pytest.fixture
def sklearn_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression()


@pytest.fixture
def uv_project(tmp_path):
    """Create a minimal UV project structure."""
    # Create uv.lock
    uv_lock_content = """version = 1
requires-python = ">=3.10"

[[package]]
name = "numpy"
version = "1.24.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "scikit-learn"
version = "1.2.0"
source = { registry = "https://pypi.org/simple" }
"""
    (tmp_path / _UV_LOCK_FILE).write_text(uv_lock_content)

    # Create pyproject.toml
    pyproject_content = """[project]
name = "test-uv-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "scikit-learn>=1.2.0",
]
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)

    # Create .python-version
    (tmp_path / _PYTHON_VERSION_FILE).write_text("3.11.5")

    return tmp_path


@pytest.fixture
def uv_project_no_python_version(tmp_path):
    """Create a UV project without .python-version file."""
    (tmp_path / _UV_LOCK_FILE).write_text('version = 1\nrequires-python = ">=3.10"\n')
    pyproject_content = """[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.11"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)
    return tmp_path


# --- Model Logging Tests (pyfunc) ---
# Note: UV file copying is implemented in mlflow.pyfunc.log_model/save_model


def test_pyfunc_log_model_copies_uv_artifacts(uv_project, python_model, monkeypatch):
    monkeypatch.chdir(uv_project)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
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


def test_pyfunc_log_model_uses_uv_python_version_in_python_env(
    uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
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


def test_pyfunc_log_model_extracts_python_version_from_pyproject(
    uv_project_no_python_version, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project_no_python_version)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
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


def test_pyfunc_log_model_respects_mlflow_log_uv_files_env_var(
    uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project)
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", "false")

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # UV artifacts should NOT be copied
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()

        # But requirements.txt should still exist (from UV export)
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


def test_pyfunc_log_model_with_explicit_uv_lock_parameter(
    tmp_path, uv_project, python_model, monkeypatch
):
    # Work from a different directory than the UV project
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
        # Use explicit uv_lock parameter to point to UV project
        mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            uv_lock=uv_project / _UV_LOCK_FILE,
        )

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify UV artifacts from explicit path are copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert "test-uv-project" in (artifact_dir / _PYPROJECT_FILE).read_text()


# --- Fallback Tests ---


def test_pyfunc_log_model_falls_back_when_uv_not_available(uv_project, python_model, monkeypatch):
    monkeypatch.chdir(uv_project)

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


def test_pyfunc_log_model_falls_back_when_uv_export_fails(uv_project, python_model, monkeypatch):
    monkeypatch.chdir(uv_project)

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


def test_load_pyfunc_model_with_uv_artifacts(uv_project, python_model, monkeypatch):
    monkeypatch.chdir(uv_project)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
        mlflow.pyfunc.log_model(name="model", python_model=python_model)
        model_uri = f"runs:/{run.info.run_id}/model"

        # Load as pyfunc
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        assert loaded_model is not None
        # Verify model info contains expected metadata
        assert loaded_model.metadata is not None


def test_load_pyfunc_model_can_predict_after_loading(uv_project, python_model, monkeypatch):
    monkeypatch.chdir(uv_project)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mlflow.start_run() as run,
    ):
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


def test_pyfunc_save_model_with_uv_project(uv_project, python_model, tmp_path, monkeypatch):
    monkeypatch.chdir(uv_project)
    model_path = tmp_path / "saved_model"

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        mlflow.pyfunc.save_model(model_path, python_model=python_model)

        # Verify artifacts
        assert (model_path / _REQUIREMENTS_FILE_NAME).exists()
        assert (model_path / _UV_LOCK_FILE).exists()
        assert (model_path / _PYPROJECT_FILE).exists()
        assert (model_path / _PYTHON_VERSION_FILE).exists()


def test_pyfunc_save_model_with_explicit_uv_lock(uv_project, python_model, tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    model_path = tmp_path / "saved_model"
    monkeypatch.chdir(work_dir)

    uv_export_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        mlflow.pyfunc.save_model(
            model_path,
            python_model=python_model,
            uv_lock=uv_project / _UV_LOCK_FILE,
        )

        # Verify UV artifacts from explicit path
        assert (model_path / _UV_LOCK_FILE).exists()
        assert (model_path / _PYPROJECT_FILE).exists()


# --- Environment Marker Filtering Tests ---


def test_pyfunc_log_model_filters_platform_specific_dependencies(
    uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(uv_project)

    # UV export output with platform-specific packages
    uv_export_output = """numpy==1.24.0
pywin32==306 ; sys_platform == 'win32'
pandas==2.0.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_export_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
        mock.patch("sys.platform", "darwin"),  # Simulate macOS
        mlflow.start_run() as run,
    ):
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        requirements_content = (artifact_dir / _REQUIREMENTS_FILE_NAME).read_text()

        # pywin32 should be filtered out on non-Windows
        assert "pywin32" not in requirements_content
        assert "numpy" in requirements_content.lower()
