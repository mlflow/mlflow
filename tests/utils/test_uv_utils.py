import subprocess
from unittest import mock

import pytest
from packaging.version import Version

from mlflow.environment_variables import MLFLOW_UV_AUTO_DETECT
from mlflow.utils.environment import infer_pip_requirements
from mlflow.utils.uv_utils import (
    _PYPROJECT_FILE,
    _UV_LOCK_FILE,
    copy_uv_project_files,
    create_uv_sync_pyproject,
    detect_uv_project,
    export_uv_requirements,
    extract_index_urls_from_uv_lock,
    get_uv_version,
    has_uv_lock_artifact,
    is_uv_available,
    run_uv_sync,
    setup_uv_sync_environment,
)

# --- get_uv_version tests ---


def test_get_uv_version_returns_none_when_uv_not_installed():
    with mock.patch("mlflow.utils.uv_utils.shutil.which", return_value=None):
        assert get_uv_version() is None


def test_get_uv_version_returns_version_when_uv_installed():
    mock_result = mock.Mock()
    mock_result.stdout = "uv 0.5.0 (abc123 2024-01-01)"
    with (
        mock.patch("mlflow.utils.uv_utils.shutil.which", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result) as mock_run,
    ):
        version = get_uv_version()
        assert version == Version("0.5.0")
        mock_run.assert_called_once()


def test_get_uv_version_returns_none_on_subprocess_error():
    with (
        mock.patch("mlflow.utils.uv_utils.shutil.which", return_value="/usr/bin/uv"),
        mock.patch(
            "mlflow.utils.uv_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv"),
        ),
    ):
        assert get_uv_version() is None


def test_get_uv_version_returns_none_on_parse_error():
    mock_result = mock.Mock()
    mock_result.stdout = "invalid output"
    with (
        mock.patch("mlflow.utils.uv_utils.shutil.which", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        assert get_uv_version() is None


# --- is_uv_available tests ---


def test_is_uv_available_returns_false_when_uv_not_installed():
    with mock.patch("mlflow.utils.uv_utils.shutil.which", return_value=None):
        assert is_uv_available() is False


def test_is_uv_available_returns_false_when_version_below_minimum():
    mock_result = mock.Mock()
    mock_result.stdout = "uv 0.4.0 (abc123 2024-01-01)"
    with (
        mock.patch("mlflow.utils.uv_utils.shutil.which", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        assert is_uv_available() is False


@pytest.mark.parametrize("version_str", ["0.5.0", "1.0.0"])
def test_is_uv_available_returns_true_when_version_meets_or_exceeds_minimum(version_str):
    mock_result = mock.Mock()
    mock_result.stdout = f"uv {version_str} (abc123 2024-01-01)"
    with (
        mock.patch("mlflow.utils.uv_utils.shutil.which", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        assert is_uv_available() is True


# --- detect_uv_project tests ---


@pytest.mark.parametrize(
    ("create_uv_lock", "create_pyproject"),
    [(False, True), (True, False)],
    ids=["missing_uv_lock", "missing_pyproject"],
)
def test_detect_uv_project_returns_none_when_file_missing(
    tmp_path, create_uv_lock, create_pyproject
):
    if create_uv_lock:
        (tmp_path / _UV_LOCK_FILE).touch()
    if create_pyproject:
        (tmp_path / _PYPROJECT_FILE).touch()
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
    with mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value=None):
        assert export_uv_requirements() is None


def test_export_uv_requirements_returns_requirements_list(tmp_path):
    uv_output = """requests==2.28.0
numpy==1.24.0
pandas==2.0.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result) as mock_run,
    ):
        result = export_uv_requirements(tmp_path)

        assert result == ["requests==2.28.0", "numpy==1.24.0", "pandas==2.0.0"]
        mock_run.assert_called_once()


def test_export_uv_requirements_preserves_environment_markers(tmp_path):
    uv_output = """requests==2.28.0
pywin32==306 ; sys_platform == 'win32'
numpy==1.24.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        assert len(result) == 3
        assert "pywin32==306 ; sys_platform == 'win32'" in result


def test_export_uv_requirements_keeps_all_marker_variants(tmp_path):
    uv_output = """numpy==2.2.6 ; python_version < '3.11'
numpy==2.4.1 ; python_version >= '3.11'
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        numpy_entries = [r for r in result if r.startswith("numpy")]
        assert len(numpy_entries) == 2


def test_export_uv_requirements_returns_none_on_subprocess_error(tmp_path):
    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch(
            "mlflow.utils.uv_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv"),
        ),
    ):
        assert export_uv_requirements(tmp_path) is None


def test_export_uv_requirements_with_explicit_directory(tmp_path):
    (tmp_path / _UV_LOCK_FILE).touch()

    uv_output = """requests==2.28.0
numpy==1.24.0
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result) as mock_run,
    ):
        result = export_uv_requirements(directory=tmp_path)

        assert result is not None
        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cwd"] == tmp_path


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
    assert (dest_dir / _UV_LOCK_FILE).read_text() == "lock content"
    assert (dest_dir / _PYPROJECT_FILE).read_text() == "pyproject content"


@pytest.mark.parametrize(
    ("has_python_version", "expected_exists"),
    [(True, True), (False, False)],
    ids=["with_python_version", "without_python_version"],
)
def test_copy_uv_project_files_python_version_handling(
    tmp_path, has_python_version, expected_exists
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / _UV_LOCK_FILE).write_text("lock content")
    (source_dir / _PYPROJECT_FILE).write_text("pyproject content")
    if has_python_version:
        (source_dir / ".python-version").write_text("3.11.5")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir)

    assert result is True
    assert (dest_dir / ".python-version").exists() == expected_exists
    if expected_exists:
        assert (dest_dir / ".python-version").read_text() == "3.11.5"


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


@pytest.mark.parametrize("env_value", ["0", "false", "FALSE", "False"])
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


@pytest.mark.parametrize("env_value", ["true", "1", "TRUE", "True"])
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


def test_copy_uv_project_files_with_monorepo_layout(tmp_path):
    project_dir = tmp_path / "monorepo" / "subproject"
    project_dir.mkdir(parents=True)
    (project_dir / _UV_LOCK_FILE).write_text("lock content from monorepo")
    (project_dir / _PYPROJECT_FILE).write_text("pyproject from monorepo")
    (project_dir / ".python-version").write_text("3.12.0")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir=project_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).read_text() == "lock content from monorepo"
    assert (dest_dir / ".python-version").read_text() == "3.12.0"


def test_copy_uv_project_files_with_nonexistent_source(tmp_path):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    nonexistent_dir = tmp_path / "nonexistent"

    result = copy_uv_project_files(dest_dir, source_dir=nonexistent_dir)
    assert result is False


def test_copy_uv_project_files_with_missing_pyproject(tmp_path):
    project_dir = tmp_path / "incomplete_project"
    project_dir.mkdir()
    (project_dir / _UV_LOCK_FILE).write_text("lock content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir, source_dir=project_dir)
    assert result is False


# --- Integration tests for infer_pip_requirements uv path ---


def test_infer_pip_requirements_uses_uv_when_project_detected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    uv_output = "requests==2.28.0\nnumpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = infer_pip_requirements("runs:/fake/model", "sklearn")

        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result


def test_export_uv_requirements_returns_none_when_uv_binary_missing(tmp_path):
    with mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value=None):
        result = export_uv_requirements(tmp_path)
        assert result is None


def test_detect_uv_project_not_detected_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert detect_uv_project() is None


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
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").touch()
    (tmp_path / "pyproject.toml").touch()

    assert detect_uv_project() is not None

    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "false")

    with (
        mock.patch("mlflow.utils.environment.detect_uv_project") as mock_detect,
        mock.patch("mlflow.utils.environment.export_uv_requirements") as mock_export,
        mock.patch(
            "mlflow.utils.environment._infer_requirements",
            return_value=["scikit-learn==1.0"],
        ),
    ):
        result = infer_pip_requirements(str(tmp_path), "sklearn")

        mock_detect.assert_not_called()
        mock_export.assert_not_called()
        assert "scikit-learn==1.0" in result


def test_infer_pip_requirements_uses_explicit_uv_project_dir(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")

    uv_project = tmp_path / "my_project"
    uv_project.mkdir()
    (uv_project / _UV_LOCK_FILE).touch()
    (uv_project / _PYPROJECT_FILE).touch()

    uv_output = "requests==2.28.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = infer_pip_requirements(str(tmp_path), "sklearn", uv_project_dir=uv_project)

        assert "requests==2.28.0" in result


def test_infer_pip_requirements_explicit_uv_project_dir_overrides_disabled_auto_detect(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "false")

    uv_project = tmp_path / "my_project"
    uv_project.mkdir()
    (uv_project / _UV_LOCK_FILE).touch()
    (uv_project / _PYPROJECT_FILE).touch()

    uv_output = "numpy==1.24.0\n"
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = infer_pip_requirements(str(tmp_path), "sklearn", uv_project_dir=uv_project)

        assert "numpy==1.24.0" in result


def test_export_uv_requirements_strips_comment_lines(tmp_path):
    uv_output = """requests==2.28.0
# via
#   some-package
urllib3==1.26.0
# via requests
certifi==2023.7.22
"""
    mock_result = mock.Mock()
    mock_result.stdout = uv_output

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result == ["requests==2.28.0", "urllib3==1.26.0", "certifi==2023.7.22"]


def test_export_uv_requirements_returns_empty_list_on_empty_output(tmp_path):
    mock_result = mock.Mock()
    mock_result.stdout = ""

    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run", return_value=mock_result),
    ):
        result = export_uv_requirements(tmp_path)

        assert result == []


# --- Private Index URL Extraction Tests ---


@pytest.mark.parametrize(
    ("uv_lock_content", "expected_urls"),
    [
        (
            """
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
""",
            ["https://internal.company.com/simple"],
        ),
        (
            """
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
""",
            ["https://private1.com/simple", "https://private2.com/simple"],
        ),
        (
            """
version = 1

[[package]]
name = "numpy"
source = { registry = "https://pypi.org/simple" }
""",
            [],
        ),
    ],
    ids=["single_private", "multiple_private_deduped", "no_private"],
)
def test_extract_index_urls_from_uv_lock(tmp_path, uv_lock_content, expected_urls):
    uv_lock_path = tmp_path / "uv.lock"
    uv_lock_path.write_text(uv_lock_content)

    result = extract_index_urls_from_uv_lock(uv_lock_path)
    assert result == expected_urls


def test_extract_index_urls_from_uv_lock_file_not_exists(tmp_path):
    result = extract_index_urls_from_uv_lock(tmp_path / "nonexistent.lock")
    assert result == []


# --- uv Sync Environment Setup Tests ---


@pytest.mark.parametrize(
    ("python_version", "project_name", "expected_name", "expected_python"),
    [
        ("3.11.5", "mlflow-model-env", "mlflow-model-env", "==3.11.5"),
        ("3.10.14", "my-custom-env", "my-custom-env", "==3.10.14"),
    ],
    ids=["default_name", "custom_name"],
)
def test_create_uv_sync_pyproject(
    tmp_path, python_version, project_name, expected_name, expected_python
):
    result_path = create_uv_sync_pyproject(tmp_path, python_version, project_name=project_name)

    assert result_path.exists()
    content = result_path.read_text()
    assert f'name = "{expected_name}"' in content
    assert f'requires-python = "{expected_python}"' in content


def test_setup_uv_sync_environment(tmp_path):
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


def test_setup_uv_sync_environment_copies_existing_pyproject(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    original_pyproject = '[project]\nname = "my-model"\nversion = "1.0.0"\n'
    (model_path / "uv.lock").write_text('version = 1\nrequires-python = ">=3.11"')
    (model_path / "pyproject.toml").write_text(original_pyproject)

    env_dir = tmp_path / "env"

    result = setup_uv_sync_environment(env_dir, model_path, "3.11.5")

    assert result is True
    # Copied from model, not generated (should have "my-model" not "mlflow-model-env")
    pyproject_content = (env_dir / "pyproject.toml").read_text()
    assert 'name = "my-model"' in pyproject_content
    assert "mlflow-model-env" not in pyproject_content


def test_setup_uv_sync_environment_no_uv_lock(tmp_path):
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
    with mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value=None):
        result = run_uv_sync(tmp_path)
        assert result is False


def test_run_uv_sync_builds_correct_command(tmp_path):
    with (
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch("mlflow.utils.uv_utils.subprocess.run") as mock_run,
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
        mock.patch("mlflow.utils.uv_utils._get_uv_binary", return_value="/usr/bin/uv"),
        mock.patch(
            "mlflow.utils.uv_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv sync"),
        ),
    ):
        result = run_uv_sync(tmp_path)
        assert result is False
