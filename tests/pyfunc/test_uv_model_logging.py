"""
Integration tests for uv package manager support in model logging and loading.

Tests the end-to-end workflow:
1. uv project detection during log_model()
2. Artifact generation (uv.lock, pyproject.toml, .python-version, requirements.txt)
3. Model loading with uv artifacts

These tests use REAL uv calls (not mocked) where possible, following MLflow best practices.
Tests requiring uv are skipped if uv is not installed or below minimum version.
"""

import platform
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

# Skip marker for tests requiring uv
requires_uv = pytest.mark.skipif(
    not is_uv_available(),
    reason="uv is not installed or below minimum required version (0.5.0)",
)


class SimplePythonModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture
def python_model():
    return SimplePythonModel()


@pytest.fixture
def tmp_uv_project(tmp_path):
    """Create a real uv project with uv lock."""
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
    (pkg_dir / "__init__.py").write_text('"""Test uv project."""\n__version__ = "0.1.0"\n')

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


# --- Model Logging Tests with Real uv ---


@requires_uv
def test_pyfunc_log_model_copies_uv_artifacts(tmp_uv_project, python_model, monkeypatch):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # Verify uv artifacts are copied
        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert (artifact_dir / _PYTHON_VERSION_FILE).exists()

        # Verify content matches source
        assert "version = 1" in (artifact_dir / _UV_LOCK_FILE).read_text()
        assert "test_uv_project" in (artifact_dir / _PYPROJECT_FILE).read_text()
        assert "3.11.5" in (artifact_dir / _PYTHON_VERSION_FILE).read_text()


@requires_uv
def test_pyfunc_log_model_python_env_uses_current_python_version(
    tmp_uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        python_env_file = artifact_dir / _PYTHON_ENV_FILE_NAME
        assert python_env_file.exists()
        python_env_content = python_env_file.read_text()
        # python_env.yaml always uses the current interpreter version
        assert platform.python_version() in python_env_content


@requires_uv
def test_pyfunc_log_model_respects_mlflow_log_uv_files_env_var(
    tmp_uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", "false")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        # uv artifacts should NOT be copied when env var is false
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()

        # But requirements.txt should still exist (from uv export)
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()
        requirements_content = (artifact_dir / _REQUIREMENTS_FILE_NAME).read_text()
        assert "numpy" in requirements_content.lower()


@requires_uv
def test_pyfunc_log_model_with_explicit_uv_project_path_parameter(
    tmp_path, tmp_uv_project, python_model, monkeypatch
):
    # Work from a different directory than the uv project
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            uv_project_path=tmp_uv_project,
        )

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()
        assert "test_uv_project" in (artifact_dir / _PYPROJECT_FILE).read_text()


@requires_uv
def test_pyfunc_log_model_generates_requirements_from_uv_export(
    tmp_uv_project, python_model, monkeypatch
):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        requirements_file = artifact_dir / _REQUIREMENTS_FILE_NAME
        assert requirements_file.exists()
        requirements_content = requirements_file.read_text()
        assert "numpy" in requirements_content.lower()


# --- Fallback Tests (mocking required to simulate uv unavailable) ---


def test_pyfunc_log_model_falls_back_when_uv_not_available(tmp_path, python_model, monkeypatch):
    (tmp_path / _UV_LOCK_FILE).write_text('version = 1\nrequires-python = ">=3.10"\n')
    (tmp_path / _PYPROJECT_FILE).write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
    monkeypatch.chdir(tmp_path)

    with mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value=None):
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(name="model", python_model=python_model)

            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="model"
            )
            artifact_dir = Path(artifact_path)

            assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


def test_pyfunc_log_model_falls_back_when_uv_export_fails(tmp_path, python_model, monkeypatch):
    (tmp_path / _UV_LOCK_FILE).write_text('version = 1\nrequires-python = ">=3.10"\n')
    (tmp_path / _PYPROJECT_FILE).write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch(
            "mlflow.utils.uv_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv"),
        ),
    ):
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(name="model", python_model=python_model)

            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="model"
            )
            artifact_dir = Path(artifact_path)

            assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


def test_pyfunc_log_model_non_uv_project_uses_standard_inference(
    python_model, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()
        assert (artifact_dir / _PYTHON_ENV_FILE_NAME).exists()
        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()


# --- Model Loading Tests ---


@requires_uv
def test_load_pyfunc_model_with_uv_artifacts_and_predict(tmp_uv_project, python_model, monkeypatch):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)
        model_uri = f"runs:/{run.info.run_id}/model"

        loaded_model = mlflow.pyfunc.load_model(model_uri)

        assert loaded_model is not None
        assert loaded_model.metadata is not None

        import pandas as pd

        test_input = pd.DataFrame({"a": [1, 2, 3]})
        predictions = loaded_model.predict(test_input)
        assert predictions is not None


# --- Save Model Tests ---


@requires_uv
def test_pyfunc_save_model_with_uv_project(tmp_uv_project, python_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    model_path = tmp_path / "saved_model"

    mlflow.pyfunc.save_model(model_path, python_model=python_model)

    assert (model_path / _REQUIREMENTS_FILE_NAME).exists()
    assert (model_path / _UV_LOCK_FILE).exists()
    assert (model_path / _PYPROJECT_FILE).exists()
    assert (model_path / _PYTHON_VERSION_FILE).exists()


@requires_uv
def test_pyfunc_save_model_with_explicit_uv_project_path(
    tmp_uv_project, python_model, tmp_path, monkeypatch
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    model_path = tmp_path / "saved_model"
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    mlflow.pyfunc.save_model(
        model_path,
        python_model=python_model,
        uv_project_path=tmp_uv_project,
    )

    assert (model_path / _UV_LOCK_FILE).exists()
    assert (model_path / _PYPROJECT_FILE).exists()


# --- Environment Variable Variations ---


@requires_uv
@pytest.mark.parametrize("env_value", ["false", "0", "FALSE", "False"])
def test_mlflow_log_uv_files_env_var_false_variants(
    tmp_uv_project, python_model, monkeypatch, env_value
):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        assert not (artifact_dir / _UV_LOCK_FILE).exists()
        assert not (artifact_dir / _PYPROJECT_FILE).exists()
        assert (artifact_dir / _REQUIREMENTS_FILE_NAME).exists()


@requires_uv
@pytest.mark.parametrize("env_value", ["true", "1", "TRUE", "True"])
def test_mlflow_log_uv_files_env_var_true_variants(
    tmp_uv_project, python_model, monkeypatch, env_value
):
    monkeypatch.chdir(tmp_uv_project)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=python_model)

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model"
        )
        artifact_dir = Path(artifact_path)

        assert (artifact_dir / _UV_LOCK_FILE).exists()
        assert (artifact_dir / _PYPROJECT_FILE).exists()


# --- Dependency Groups Integration Tests ---


@pytest.fixture
def uv_project_with_groups(tmp_path):
    """Create a uv project with dependency groups."""
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

    pkg_dir = tmp_path / "test_uv_groups"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Test uv groups project."""\n__version__ = "0.1.0"\n')

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
    assert "numpy" in pkg_names
    assert "gunicorn" in pkg_names
    assert "pytest" not in pkg_names


@requires_uv
def test_export_uv_requirements_with_only_groups_real(uv_project_with_groups):
    from mlflow.utils.uv_utils import export_uv_requirements

    result = export_uv_requirements(uv_project_with_groups, only_groups=["serving"])

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    assert "gunicorn" in pkg_names


@requires_uv
def test_export_uv_requirements_with_extras_real(uv_project_with_groups):
    from mlflow.utils.uv_utils import export_uv_requirements

    result = export_uv_requirements(uv_project_with_groups, extras=["gpu"])

    assert result is not None
    pkg_names = [r.split("==")[0].lower() for r in result]
    assert "numpy" in pkg_names
    assert "scipy" in pkg_names


# --- uv Sync Environment Setup Integration Tests ---


@requires_uv
def test_setup_uv_sync_environment_real(tmp_uv_project, tmp_path):
    from mlflow.utils.uv_utils import has_uv_lock_artifact, setup_uv_sync_environment

    model_artifacts = tmp_path / "model_artifacts"
    model_artifacts.mkdir()

    shutil.copy2(tmp_uv_project / _UV_LOCK_FILE, model_artifacts / _UV_LOCK_FILE)
    shutil.copy2(tmp_uv_project / _PYTHON_VERSION_FILE, model_artifacts / _PYTHON_VERSION_FILE)

    assert has_uv_lock_artifact(model_artifacts)

    env_dir = tmp_path / "env"
    result = setup_uv_sync_environment(env_dir, model_artifacts, "3.11.5")

    assert result is True
    assert (env_dir / _UV_LOCK_FILE).exists()
    assert (env_dir / _PYPROJECT_FILE).exists()
    assert (env_dir / _PYTHON_VERSION_FILE).exists()

    # No pyproject.toml in model_artifacts, so create_uv_sync_pyproject
    # generates one with pinned version
    pyproject_content = (env_dir / _PYPROJECT_FILE).read_text()
    assert 'name = "mlflow-model-env"' in pyproject_content
    assert 'requires-python = "==3.11.5"' in pyproject_content


@requires_uv
def test_extract_index_urls_from_real_uv_lock(tmp_uv_project):
    from mlflow.utils.uv_utils import extract_index_urls_from_uv_lock

    result = extract_index_urls_from_uv_lock(tmp_uv_project / _UV_LOCK_FILE)

    well_known_public = {"https://download.pytorch.org/whl/cpu"}
    truly_private = [url for url in result if url not in well_known_public]
    assert truly_private == []


@requires_uv
def test_run_uv_sync_real(tmp_uv_project, tmp_path):
    from mlflow.utils.uv_utils import run_uv_sync

    sync_dir = tmp_path / "sync_project"
    shutil.copytree(tmp_uv_project, sync_dir)

    result = run_uv_sync(sync_dir, frozen=True, no_dev=True)

    assert result is True
    assert (sync_dir / ".venv").exists()
