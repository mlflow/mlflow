import subprocess
import sys
from pathlib import Path

import pytest


def get_check_init_py_script() -> Path:
    return Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"


def get_precommit_config() -> Path:
    return Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"


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


def test_precommit_hook_file_filter_pattern() -> None:
    """Test that the pre-commit hook configuration has correct file filtering."""
    import yaml

    config_path = get_precommit_config()
    assert config_path.exists(), f"Pre-commit config not found at {config_path}"

    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Find the check-init-py hook
    check_init_hook = None
    for repo in config.get("repos", []):
        if repo.get("repo") == "local":
            for hook in repo.get("hooks", []):
                if hook.get("id") == "check-init-py":
                    check_init_hook = hook
                    break
            if check_init_hook:
                break

    assert check_init_hook is not None, "check-init-py hook not found in .pre-commit-config.yaml"
    assert "files" in check_init_hook, "check-init-py hook should have a 'files' pattern"
    assert check_init_hook["files"] == "^mlflow/.*\\.py$", (
        f"Expected files pattern '^mlflow/.*\\.py$', got '{check_init_hook['files']}'"
    )
