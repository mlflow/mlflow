import subprocess
import sys
from pathlib import Path


def test_check_init_py_no_missing_files(tmp_path, monkeypatch):
    """Test that the script exits with 0 when all directories have __init__.py files."""
    # Create test directory structure with proper __init__.py files
    mlflow_dir = tmp_path / "mlflow"
    test_package_dir = mlflow_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    # Create __init__.py files
    (mlflow_dir / "__init__.py").touch()
    (test_package_dir / "__init__.py").touch()

    # Create a test Python file
    (test_package_dir / "test_module.py").touch()

    # Initialize git repo and add files
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Change to test directory
    monkeypatch.chdir(tmp_path)

    # Run the script - get absolute path from test file location
    script_path = Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    assert result.returncode == 0
    assert result.stdout == ""


def test_check_init_py_missing_files(tmp_path, monkeypatch):
    """Test that the script exits with 1 when directories are missing __init__.py files."""
    # Create test directory structure without __init__.py files
    mlflow_dir = tmp_path / "mlflow"
    test_package_dir = mlflow_dir / "test_package"
    test_package_dir.mkdir(parents=True)

    # Create a test Python file but no __init__.py files
    (test_package_dir / "test_module.py").touch()

    # Initialize git repo and add files
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Change to test directory
    monkeypatch.chdir(tmp_path)

    # Run the script
    script_path = Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    assert "mlflow" in result.stdout
    assert "mlflow/test_package" in result.stdout


def test_check_init_py_no_python_files(tmp_path, monkeypatch):
    """Test that the script exits with 0 when no Python files are found."""
    # Create test directory structure with no Python files
    mlflow_dir = tmp_path / "mlflow"
    js_dir = mlflow_dir / "server" / "js"
    js_dir.mkdir(parents=True)

    # Create a non-Python file
    (js_dir / "main.js").touch()

    # Initialize git repo and add files
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Change to test directory
    monkeypatch.chdir(tmp_path)

    # Run the script
    script_path = Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    assert result.returncode == 0
    assert result.stdout == ""


def test_check_init_py_partial_missing(tmp_path, monkeypatch):
    """Test that the script correctly identifies only the directories missing __init__.py."""
    # Create test directory structure
    mlflow_dir = tmp_path / "mlflow"
    package1_dir = mlflow_dir / "package1"
    package2_dir = mlflow_dir / "package2"
    package1_dir.mkdir(parents=True)
    package2_dir.mkdir(parents=True)

    # Create __init__.py for mlflow and package1, but not package2
    (mlflow_dir / "__init__.py").touch()
    (package1_dir / "__init__.py").touch()
    # package2 intentionally missing __init__.py

    # Create Python files in both packages
    (package1_dir / "module1.py").touch()
    (package2_dir / "module2.py").touch()

    # Initialize git repo and add files
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Change to test directory
    monkeypatch.chdir(tmp_path)

    # Run the script
    script_path = Path(__file__).resolve().parents[2] / "dev" / "check_init_py.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    assert result.returncode == 1
    assert (
        "Error: The following directories contain Python files but lack __init__.py:"
        in result.stdout
    )
    assert "mlflow/package2" in result.stdout
    assert "mlflow/package1" not in result.stdout  # This should not be listed as it has __init__.py
