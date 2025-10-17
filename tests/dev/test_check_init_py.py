import subprocess
import sys
from pathlib import Path

import pytest


def get_check_init_py_script() -> Path:
    return Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    return tmp_path


def test_exits_with_0_when_all_directories_have_init_py(temp_git_repo: Path) -> None:
    mlflow_dir = temp_git_repo / "mlflow"
    test_package_dir = mlflow_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    (mlflow_dir / "__init__.py").touch()
    (test_package_dir / "__init__.py").touch()
    (test_package_dir / "test_module.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_exits_with_1_when_directories_missing_init_py(temp_git_repo: Path) -> None:
    mlflow_dir = temp_git_repo / "mlflow"
    test_package_dir = mlflow_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    (test_package_dir / "test_module.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    assert "mlflow" in result.stdout
    assert "mlflow/test_package" in result.stdout


def test_exits_with_0_when_no_python_files_exist(temp_git_repo: Path) -> None:
    mlflow_dir = temp_git_repo / "mlflow"
    js_dir = mlflow_dir / "server" / "js"
    js_dir.mkdir(parents=True)

    (js_dir / "main.js").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_identifies_only_directories_missing_init_py(temp_git_repo: Path) -> None:
    mlflow_dir = temp_git_repo / "mlflow"
    package1_dir = mlflow_dir / "package1"
    package2_dir = mlflow_dir / "package2"
    package1_dir.mkdir(parents=True)
    package2_dir.mkdir(parents=True)

    (mlflow_dir / "__init__.py").touch()
    (package1_dir / "__init__.py").touch()

    (package1_dir / "module1.py").touch()
    (package2_dir / "module2.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    assert "mlflow/package2" in result.stdout
    assert "mlflow/package1" not in result.stdout


def test_checks_tests_directory_for_missing_init_py(temp_git_repo: Path) -> None:
    tests_dir = temp_git_repo / "tests"
    test_package_dir = tests_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    # Only test files (starting with test_) are checked
    (test_package_dir / "test_module.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    assert "tests/test_package" in result.stdout
    assert "tests" in result.stdout  # Parent directory also needs __init__.py


def test_exits_with_0_when_tests_directories_have_init_py(temp_git_repo: Path) -> None:
    tests_dir = temp_git_repo / "tests"
    test_package_dir = tests_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    (tests_dir / "__init__.py").touch()
    (test_package_dir / "__init__.py").touch()
    (test_package_dir / "test_module.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_ignores_non_test_files_in_tests_directory(temp_git_repo: Path) -> None:
    tests_dir = temp_git_repo / "tests"
    test_package_dir = tests_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    # Non-test file (doesn't start with test_) should be ignored
    (test_package_dir / "helper.py").touch()
    (test_package_dir / "utils.py").touch()

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    # Should pass since no test files exist
    assert result.returncode == 0
    assert result.stdout == ""


def test_checks_all_parent_directories(temp_git_repo: Path) -> None:
    mlflow_dir = temp_git_repo / "mlflow"
    deep_dir = mlflow_dir / "level1" / "level2" / "level3"
    deep_dir.mkdir(parents=True)

    # Create a Python file deep in the hierarchy
    (deep_dir / "module.py").touch()

    # Only add __init__.py to some directories
    (mlflow_dir / "__init__.py").touch()
    (mlflow_dir / "level1" / "__init__.py").touch()
    # Missing: level2 and level3 __init__.py files

    subprocess.check_call(["git", "add", "."], cwd=temp_git_repo)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=temp_git_repo)

    result = subprocess.run(
        [sys.executable, get_check_init_py_script()],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    # Both level2 and level3 should be reported as missing __init__.py
    assert "mlflow/level1/level2" in result.stdout
    assert "mlflow/level1/level2/level3" in result.stdout
