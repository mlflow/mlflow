import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Import the module we're testing
sys.path.insert(0, str(Path(__file__).parent.parent))
from dev.check_init_py import (
    check_missing_init_files,
    get_python_directories,
    get_tracked_python_files,
    main,
)


def test_get_python_directories():
    """Test extraction of directories from Python file paths."""
    python_files = [
        "mlflow/__init__.py",
        "mlflow/tracking/client.py",
        "mlflow/models/model.py",
        "mlflow/utils/file_utils.py",
        "mlflow/utils/logging_utils.py",
    ]

    directories = get_python_directories(python_files)
    expected = {
        "mlflow/tracking",
        "mlflow/models",
        "mlflow/utils",
    }

    assert directories == expected


def test_get_python_directories_excludes_root():
    """Test that root mlflow directory is excluded."""
    python_files = ["mlflow/__init__.py", "mlflow/client.py"]
    directories = get_python_directories(python_files)

    # Should not include "mlflow" directory itself
    assert "mlflow" not in directories
    assert len(directories) == 0


def test_check_missing_init_files(tmp_path, monkeypatch):
    """Test detection of missing __init__.py files."""
    # Create test directory structure
    test_dir1 = tmp_path / "mlflow" / "test_package1"
    test_dir2 = tmp_path / "mlflow" / "test_package2"
    test_dir1.mkdir(parents=True)
    test_dir2.mkdir(parents=True)

    # Create __init__.py in one directory but not the other
    (test_dir1 / "__init__.py").touch()
    # test_dir2 intentionally missing __init__.py

    # Change to tmpdir to test relative paths
    monkeypatch.chdir(tmp_path)

    directories = {"mlflow/test_package1", "mlflow/test_package2"}
    missing = check_missing_init_files(directories)

    assert missing == ["mlflow/test_package2"]


def test_check_missing_init_files_all_present(tmp_path, monkeypatch):
    """Test that no missing files are reported when all __init__.py files exist."""
    # Create test directory structure
    test_dir1 = tmp_path / "mlflow" / "test_package1"
    test_dir2 = tmp_path / "mlflow" / "test_package2"
    test_dir1.mkdir(parents=True)
    test_dir2.mkdir(parents=True)

    # Create __init__.py in both directories
    (test_dir1 / "__init__.py").touch()
    (test_dir2 / "__init__.py").touch()

    monkeypatch.chdir(tmp_path)

    directories = {"mlflow/test_package1", "mlflow/test_package2"}
    missing = check_missing_init_files(directories)

    assert missing == []


@mock.patch("dev.check_init_py.subprocess.run")
def test_get_tracked_python_files(mock_run):
    """Test getting tracked Python files via git ls-files."""
    # Mock git ls-files output
    mock_run.return_value = mock.Mock(
        returncode=0,
        stdout="mlflow/__init__.py\nmlflow/client.py\nmlflow/utils/file_utils.py\nmlflow/server/js/main.js\n",
    )

    files = get_tracked_python_files()
    expected = [
        "mlflow/__init__.py",
        "mlflow/client.py",
        "mlflow/utils/file_utils.py",
        # Note: JS file should be filtered out
    ]

    assert files == expected
    mock_run.assert_called_once_with(
        ["git", "ls-files", "mlflow/"],
        capture_output=True,
        text=True,
        check=True,
    )


@mock.patch("dev.check_init_py.subprocess.run")
def test_get_tracked_python_files_git_error(mock_run):
    """Test handling of git command errors."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "git")

    with pytest.raises(SystemExit, match="1"):
        get_tracked_python_files()


@mock.patch("dev.check_init_py.get_tracked_python_files")
@mock.patch("dev.check_init_py.check_missing_init_files")
def test_main_success(mock_check_missing, mock_get_files):
    """Test main function when no missing files are found."""
    mock_get_files.return_value = ["mlflow/utils/file_utils.py"]
    mock_check_missing.return_value = []

    exit_code = main()
    assert exit_code == 0


@mock.patch("dev.check_init_py.get_tracked_python_files")
@mock.patch("dev.check_init_py.check_missing_init_files")
def test_main_failure(mock_check_missing, mock_get_files):
    """Test main function when missing files are found."""
    mock_get_files.return_value = ["mlflow/test_package/test_module.py"]
    mock_check_missing.return_value = ["mlflow/test_package"]

    exit_code = main()
    assert exit_code == 1


@mock.patch("dev.check_init_py.get_tracked_python_files")
def test_main_no_python_files(mock_get_files):
    """Test main function when no Python files are found."""
    mock_get_files.return_value = []

    exit_code = main()
    assert exit_code == 0
