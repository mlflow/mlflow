import subprocess
from unittest import mock

import pytest
from packaging.version import Version

from mlflow.utils.uv_utils import (
    _MIN_UV_VERSION,
    _PYPROJECT_FILE,
    _UV_LOCK_FILE,
    _evaluate_marker,
    copy_uv_project_files,
    detect_uv_project,
    export_uv_requirements,
    get_python_version_from_uv_project,
    get_uv_version,
    is_uv_available,
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
    assert result["uv_lock"] == tmp_path / _UV_LOCK_FILE
    assert result["pyproject"] == tmp_path / _PYPROJECT_FILE


def test_detect_uv_project_uses_cwd_when_directory_not_specified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / _UV_LOCK_FILE).touch()
    (tmp_path / _PYPROJECT_FILE).touch()

    result = detect_uv_project()
    assert result is not None
    assert result["uv_lock"] == tmp_path / _UV_LOCK_FILE


# --- _evaluate_marker tests ---


@pytest.fixture
def version_info():
    return type("VersionInfo", (), {"major": 3, "minor": 11, "micro": 5})()


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("python_version < '3.12'", True),
        ("python_version > '3.12'", False),
        ("python_version == '3.11'", True),
        ("python_version != '3.11'", False),
        ("python_version >= '3.10'", True),
        ("python_version <= '3.11'", True),
    ],
)
def test_evaluate_marker_python_version(marker, expected, version_info):
    assert _evaluate_marker(marker, version_info) == expected


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("python_full_version < '3.11.6'", True),
        ("python_full_version > '3.11.4'", True),
        ("python_full_version == '3.11.5'", True),
        ("python_full_version != '3.11.5'", False),
    ],
)
def test_evaluate_marker_python_full_version(marker, expected, version_info):
    assert _evaluate_marker(marker, version_info) == expected


def test_evaluate_marker_and_condition(version_info):
    marker = "python_version >= '3.10' and python_version < '3.12'"
    assert _evaluate_marker(marker, version_info) is True

    marker = "python_version >= '3.10' and python_version < '3.11'"
    assert _evaluate_marker(marker, version_info) is False


def test_evaluate_marker_or_condition(version_info):
    marker = "python_version == '3.10' or python_version == '3.11'"
    assert _evaluate_marker(marker, version_info) is True

    marker = "python_version == '3.9' or python_version == '3.10'"
    assert _evaluate_marker(marker, version_info) is False


def test_evaluate_marker_platform_markers(version_info):
    with mock.patch("sys.platform", "darwin"):
        assert _evaluate_marker("sys_platform == 'darwin'", version_info) is True
        assert _evaluate_marker("sys_platform == 'win32'", version_info) is False

    with mock.patch("platform.system", return_value="Darwin"):
        assert _evaluate_marker("platform_system == 'Darwin'", version_info) is True
        assert _evaluate_marker("platform_system == 'Windows'", version_info) is False


def test_evaluate_marker_unparsable_marker_returns_true(version_info):
    assert _evaluate_marker("unparsable marker", version_info) is True


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


def test_export_uv_requirements_filters_environment_markers(tmp_path):
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
        mock.patch("sys.platform", "darwin"),
    ):
        result = export_uv_requirements(tmp_path)

        assert result is not None
        assert "pywin32==306" not in result
        assert "requests==2.28.0" in result
        assert "numpy==1.24.0" in result


def test_export_uv_requirements_deduplicates_packages(tmp_path):
    uv_output = """numpy==2.2.6 ; python_version < '3.11'
numpy==2.4.1
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
        assert len(numpy_entries) == 1


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


def test_copy_uv_project_files_uses_cwd_when_source_not_specified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / _UV_LOCK_FILE).write_text("lock content")
    (tmp_path / _PYPROJECT_FILE).write_text("pyproject content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_uv_project_files(dest_dir)

    assert result is True
    assert (dest_dir / _UV_LOCK_FILE).exists()
