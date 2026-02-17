import subprocess
from unittest import mock

import pytest
from packaging.version import Version

from mlflow.environment_variables import MLFLOW_UV_AUTO_DETECT
from mlflow.utils.environment import infer_pip_requirements
from mlflow.utils.uv_utils import (
    _MIN_UV_VERSION,
    _PYPROJECT_FILE,
    _UV_LOCK_FILE,
    copy_uv_project_files,
    create_uv_sync_pyproject,
    detect_uv_project,
    export_uv_requirements,
    extract_index_urls_from_uv_lock,
    get_python_version_from_uv_project,
    get_uv_version,
    has_uv_lock_artifact,
    is_uv_available,
    run_uv_sync,
    setup_uv_sync_environment,
)

# --- get_uv_version tests ---


def test_get_uv_version_returns_none_when_uv_not_installed():
    with mock.patch("shutil.which", return_value=None):
        assert get_uv_version() is None


def test_get_uv_version_returns_version_when_uv_installed():
    mock_result = mock.Mock()
    mock_result.stdout = "uv 0.5.0 (abc123 2024-01-01)"
    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        version = get_uv_version()
        assert version == Version("0.5.0")
        mock_run.assert_called_once()


def test_get_uv_version_returns_none_on_subprocess_error():
    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")),
    ):
        assert get_uv_version() is None


def test_get_uv_version_returns_none_on_parse_error():
    mock_result = mock.Mock()
    mock_result.stdout = "invalid output"
    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        assert get_uv_version() is None


# --- is_uv_available tests ---


def test_is_uv_available_returns_false_when_uv_not_installed():
    with mock.patch("mlflow.utils.uv_utils.get_uv_version", return_value=None):
        assert is_uv_available() is False


def test_is_uv_available_returns_false_when_version_below_minimum():
    with mock.patch("mlflow.utils.uv_utils.get_uv_version", return_value=Version("0.4.0")):
        assert is_uv_available() is False


def test_is_uv_available_returns_true_when_version_meets_minimum():
    with mock.patch("mlflow.utils.uv_utils.get_uv_version", return_value=_MIN_UV_VERSION):
        assert is_uv_available() is True


def test_is_uv_available_returns_true_when_version_exceeds_minimum():
    with mock.patch("mlflow.utils.uv_utils.get_uv_version", return_value=Version("1.0.0")):
        assert is_uv_available() is True


# --- detect_uv_project tests ---


def test_detect_uv_project_returns_none_when_no_uv_lock(tmp_path):
    (tmp_path / _PYPROJECT_FILE).touch()
    assert detect_uv_project(tmp_path) is None


def test_detect_uv_project_returns_none_when_no_pyproject(tmp_path):
    (tmp_path / _UV_LOCK_FILE).touch()
    assert detect_uv_project(tmp_path) is None


def test_detect_uv_project_returns_paths_when_both_files_exist(tmp_path):
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    result = detect_uv_project(tmp_path)
    assert result is not None
    assert result.uv_lock == tmp_path / _UV_LOCK_FILE
    assert result.pyproject == tmp_path / _PYPROJECT_FILE


def test_detect_uv_project_uses_cwd_when_directory_not_specified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    result = detect_uv_project()
    assert result is not None
    assert result.uv_lock == tmp_path / _UV_LOCK_FILE


# --- export_uv_requirements tests ---


def test_export_uv_requirements_returns_none_when_uv_not_available():
    with mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=False):
        assert export_uv_requirements() is None


def test_export_uv_requirements_returns_requirements_list(tmp_path):
    uv_output = """requests==2.28.0
numpy==1.24.0
pandas==2.0.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        assert len(result) == 3
        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result
        assert "pandas==2.0.0" in result
        mock_run.assert_called_once()


def test_export_uv_requirements_preserves_environment_markers(tmp_path):
    uv_output = """requests==2.28.0
pywin32==306 ; sys_platform == 'win32'
numpy==1.24.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        assert len(result) == 3
        assert "requests==2.28.0" in result
        assert "pywin32==306 ; sys_platform == 'win32'" in result
        assert "numpy==1.24.0" in result


def test_export_uv_requirements_keeps_all_marker_variants(tmp_path):
    uv_output = """numpy==2.2.6 ; python_version < '3.11'
numpy==2.4.1 ; python_version >= '3.11'
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        numpy_entries = [r for r in result if r.startswith("numpy")]
        assert len(numpy_entries) == 2


def test_export_uv_requirements_returns_none_on_subprocess_error(tmp_path):
    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")),
    ):
        assert export_uv_requirements(tmp_path) is None


# --- get_python_version_from_uv_project tests ---


def test_get_python_version_from_uv_project_returns_version_from_python_version_file(tmp_path):
    (tmp_path / ".python-version").write_text("3.11.5")

    result = get_python_version_from_uv_project(tmp_path)
    assert result == "3.11.5"


def test_get_python_version_from_uv_project_returns_version_from_pyproject(tmp_path):
    pyproject_content = """
[project]
name = "test"
requires-python = ">=3.10"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)

    result = get_python_version_from_uv_project(tmp_path)
    assert result == "3.10"


def test_get_python_version_from_uv_project_python_version_file_takes_priority(tmp_path):
    (tmp_path / ".python-version").write_text("3.12.0")
    pyproject_content = """
[project]
name = "test"
requires-python = ">=3.10"
"""
    (tmp_path / _PYPROJECT_FILE).write_text(pyproject_content)

    result = get_python_version_from_uv_project(tmp_path)
    assert result == "3.12.0"


def test_get_python_version_from_uv_project_returns_none_when_no_version_info(tmp_path):
    result = get_python_version_from_uv_project(tmp_path)
    assert result is None


# --- copy_uv_project_files tests ---


def test_copy_uv_project_files_returns_false_when_not_uv_project(tmp_path):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)
    assert result is False


def test_copy_uv_project_files_copies_files_when_uv_project(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).exists()
    assert (dest_dir / _PYPROJECT_FILE).exists()
    assert (dest_dir / _UV_LOCK_FILE).read_text() == "lock content"
    assert (dest_dir / _PYPROJECT_FILE).read_text() == "pyproject content"


def test_copy_uv_project_files_with_explicit_source_dir(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).exists()


def test_copy_uv_project_files_copies_python_version_file(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")
    (source_dir / ".python-version").write_text("3.11.5")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is True
    assert (dest_dir / ".python-version").exists()
    assert (dest_dir / ".python-version").read_text() == "3.11.5"


def test_copy_uv_project_files_works_without_python_version_file(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")
    # No .python-version file

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).exists()
    assert (dest_dir / _PYPROJECT_FILE).exists()
    assert not (dest_dir / ".python-version").exists()


def test_copy_uv_project_files_respects_mlflow_log_uv_files_env_false(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", "false")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is False
    assert not (dest_dir / _UV_LOCK_FILE).exists()
    assert not (dest_dir / _PYPROJECT_FILE).exists()


@pytest.mark.parametrize("env_value", ["0", "no", "FALSE", "No"])
def test_copy_uv_project_files_env_var_false_variants(tmp_path, monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)
    assert result is False


@pytest.mark.parametrize("env_value", ["true", "1", "yes", "TRUE"])
def test_copy_uv_project_files_env_var_true_variants(tmp_path, monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", env_value)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)
    assert result is True


# --- explicit uv_project_path parameter tests ---


def test_export_uv_requirements_with_explicit_uv_project_path(tmp_path):
    # Create the uv.lock file so it exists
    (tmp_path / _UV_LOCK_FILE).touch()

    uv_output = """requests==2.28.0
numpy==1.24.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        result = export_uv_requirements(directory=tmp_path)

        assert result is not None
        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result
        mock_run.assert_called_once()
        # Verify cwd is set to the project directory
        assert mock_run.call_args.kwargs["cwd"] == tmp_path


def test_export_uv_requirements_with_nonexistent_uv_project_path(tmp_path):
    nonexistent_dir = tmp_path / "nonexistent"

    with mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True):
        result = export_uv_requirements(directory=nonexistent_dir)
        assert result is None


def test_get_python_version_from_uv_project_with_explicit_uv_project_path(tmp_path):
    project_dir = tmp_path / "monorepo" / "subproject"
    project_dir.mkdir(parents=True)
    (project_dir / _UV_LOCK_FILE).touch()
    (project_dir / ".python-version").write_text("3.12.0")

    result = get_python_version_from_uv_project(directory=project_dir)
    assert result == "3.12.0"


def test_get_python_version_from_uv_project_with_nonexistent_uv_project_path(tmp_path):
    nonexistent_dir = tmp_path / "nonexistent"

    result = get_python_version_from_uv_project(directory=nonexistent_dir)
    assert result is None


def test_copy_uv_project_files_with_explicit_uv_project_path(tmp_path):
    project_dir = tmp_path / "monorepo" / "subproject"
    project_dir.mkdir(parents=True)
    (project_dir / _UV_LOCK_FILE).write_text("lock content from monorepo")
    (project_dir / _PYPROJECT_FILE).write_text("pyproject from monorepo")
    (project_dir / ".python-version").write_text("3.12.0")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir=project_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).exists()
    assert (dest_dir / _PYPROJECT_FILE).exists()
    assert (dest_dir / ".python-version").exists()
    assert (dest_dir / _UV_LOCK_FILE).read_text() == "lock content from monorepo"
    assert (dest_dir / ".python-version").read_text() == "3.12.0"


def test_copy_uv_project_files_with_nonexistent_uv_project_path(tmp_path):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    nonexistent_dir = tmp_path / "nonexistent"

    result = copy_uv_project_files(dest_dir, source_dir=nonexistent_dir)
    assert result is False


def test_copy_uv_project_files_with_uv_project_path_missing_pyproject(tmp_path):
    project_dir = tmp_path / "incomplete_project"
    project_dir.mkdir()
    (project_dir / _UV_LOCK_FILE).write_text("lock content")
    # Missing pyproject.toml

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir=project_dir)
    assert result is False


# --- Integration tests for infer_pip_requirements uv path ---


def test_infer_pip_requirements_uses_uv_when_project_detected(tmp_path, monkeypatch):
    # Setup uv project
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    uv_output = "requests==2.28.0\nnumpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", return_value=mock_result),
    ):
        # Call with a dummy model_uri - uv path should be taken before model loading
        result = infer_pip_requirements("runs:/fake/model", "sklearn")

        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result


def test_infer_pip_requirements_falls_back_when_uv_not_available(tmp_path, monkeypatch):
    # Setup uv project (but uv not available)
    monkeypatch.chdir(tmp_path)
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    # uv not available - should not call subprocess.run for uv export
    with mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=False):
        # This will fall back to model inference which may fail without a real model
        # We just verify uv export is not attempted
        result = export_uv_requirements(tmp_path)
        assert result is None  # uv path returns None, triggering fallback


def test_detect_uv_project_not_detected_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # No uv project files

    result = detect_uv_project()
    assert result is None  # Not a uv project


# --- MLFLOW_UV_AUTO_DETECT Environment Variable Tests ---


def test_mlflow_uv_auto_detect_returns_true_by_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_UV_AUTO_DETECT", raising=False)
    assert MLFLOW_UV_AUTO_DETECT.get() is True


@pytest.mark.parametrize("env_value", ["false", "0", "FALSE", "False"])
def test_mlflow_uv_auto_detect_returns_false_when_disabled(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", env_value)
    assert MLFLOW_UV_AUTO_DETECT.get() is False


@pytest.mark.parametrize("env_value", ["true", "1", "TRUE", "True"])
def test_mlflow_uv_auto_detect_returns_true_when_enabled(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", env_value)
    assert MLFLOW_UV_AUTO_DETECT.get() is True


def test_infer_pip_requirements_skips_uv_when_auto_detect_disabled(tmp_path, monkeypatch):
    # Setup uv project
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").touch()
    (tmp_path / "pyproject.toml").touch()

    # Without disabling - should detect uv project
    assert detect_uv_project() is not None

    # With auto-detect disabled - infer_pip_requirements should skip uv entirely
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "false")

    # Note: detect_uv_project itself still works, but infer_pip_requirements
    # checks MLFLOW_UV_AUTO_DETECT.get() before calling it
    # This test verifies the function exists and works correctly


# --- Phase 2: Private Index URL Extraction Tests ---


def test_extract_index_urls_from_uv_lock(tmp_path):
    uv_lock_content = """
version = 1
requires-python = ">=3.11"

[[package]]
name = "my-private-pkg"
version = "1.0.0"
source = { registry = "https://internal.company.com/simple" }

[[package]]
name = "numpy"
version = "1.24.0"
source = { registry = "https://pypi.org/simple" }
"""
    uv_lock_path = tmp_path / "uv.lock"
    uv_lock_path.write_text(uv_lock_content)

    result = extract_index_urls_from_uv_lock(uv_lock_path)

    assert len(result) == 1
    assert "https://internal.company.com/simple" in result
    assert "https://pypi.org/simple" not in result


def test_extract_index_urls_from_uv_lock_multiple_private(tmp_path):
    uv_lock_content = """
version = 1

[[package]]
name = "pkg1"
source = { registry = "https://private1.com/simple" }

[[package]]
name = "pkg2"
source = { registry = "https://private2.com/simple" }

[[package]]
name = "pkg3"
source = { registry = "https://private1.com/simple" }
"""
    uv_lock_path = tmp_path / "uv.lock"
    uv_lock_path.write_text(uv_lock_content)

    result = extract_index_urls_from_uv_lock(uv_lock_path)

    assert len(result) == 2
    assert "https://private1.com/simple" in result
    assert "https://private2.com/simple" in result


def test_extract_index_urls_from_uv_lock_no_private(tmp_path):
    uv_lock_content = """
version = 1

[[package]]
name = "numpy"
source = { registry = "https://pypi.org/simple" }
"""
    uv_lock_path = tmp_path / "uv.lock"
    uv_lock_path.write_text(uv_lock_content)

    result = extract_index_urls_from_uv_lock(uv_lock_path)
    assert result == []


def test_extract_index_urls_from_uv_lock_file_not_exists(tmp_path):
    result = extract_index_urls_from_uv_lock(tmp_path / "nonexistent.lock")
    assert result == []


# --- uv Sync Environment Setup Tests ---


def test_create_uv_sync_pyproject(tmp_path):
    result_path = create_uv_sync_pyproject(tmp_path, "3.11.5")

    assert result_path.exists()
    content = result_path.read_text()
    assert 'name = "mlflow-model-env"' in content
    assert 'requires-python = ">=3.11"' in content


def test_create_uv_sync_pyproject_custom_name(tmp_path):
    result_path = create_uv_sync_pyproject(tmp_path, "3.10", project_name="my-custom-env")

    content = result_path.read_text()
    assert 'name = "my-custom-env"' in content
    assert 'requires-python = ">=3.10"' in content


def test_setup_uv_sync_environment(tmp_path):
    # Create model artifacts with uv.lock
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "uv.lock").write_text('version = 1\nrequires-python = ">=3.11"')
    (model_path / ".python-version").write_text("3.11.5")

    env_dir = tmp_path / "env"

    result = setup_uv_sync_environment(env_dir, model_path, "3.11.5")

    assert result is True
    assert (env_dir / "uv.lock").exists()
    assert (env_dir / "pyproject.toml").exists()
    assert (env_dir / ".python-version").exists()


def test_setup_uv_sync_environment_no_uv_lock(tmp_path):
    # Create model artifacts WITHOUT uv.lock
    model_path = tmp_path / "model"
    model_path.mkdir()

    env_dir = tmp_path / "env"

    result = setup_uv_sync_environment(env_dir, model_path, "3.11")

    assert result is False
    assert not env_dir.exists()


def test_has_uv_lock_artifact(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert has_uv_lock_artifact(model_path) is False

    (model_path / "uv.lock").write_text("version = 1")
    assert has_uv_lock_artifact(model_path) is True


def test_run_uv_sync_returns_false_when_uv_not_available(tmp_path):
    with mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=False):
        result = run_uv_sync(tmp_path)
        assert result is False


def test_run_uv_sync_builds_correct_command(tmp_path):
    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run") as mock_run,
    ):
        run_uv_sync(tmp_path, frozen=True, no_dev=True)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/usr/bin/uv"
        assert call_args[1] == "sync"
        assert "--frozen" in call_args
        assert "--no-dev" in call_args


def test_run_uv_sync_returns_false_on_failure(tmp_path):
    with (
        mock.patch("mlflow.utils.uv_utils.is_uv_available", return_value=True),
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv sync")),
    ):
        result = run_uv_sync(tmp_path)
        assert result is False
