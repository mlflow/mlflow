import subprocess
import tempfile
from pathlib import Path


def run_check_mlflow_ui(file_path):
    """Run the check-mlflow-ui.sh script on a file."""
    return subprocess.run(
        ["dev/check-mlflow-ui.sh", str(file_path)],
        capture_output=True,
        text=True,
    )


def test_check_mlflow_ui_detects_usage():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Run mlflow ui to start the server\n")
        f.flush()
        temp_file = Path(f.name)

    try:
        result = run_check_mlflow_ui(temp_file)
        assert result.returncode == 1, "Script should fail when 'mlflow ui' is found"
        assert "mlflow ui" in result.stdout.lower()
        assert "mlflow server" in result.stdout.lower()
    finally:
        temp_file.unlink()


def test_check_mlflow_ui_allows_mlflow_server():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Run mlflow server to start the server\n")
        f.flush()
        temp_file = Path(f.name)

    try:
        result = run_check_mlflow_ui(temp_file)
        assert result.returncode == 0, "Script should pass when 'mlflow server' is used"
    finally:
        temp_file.unlink()


def test_check_mlflow_ui_multiple_occurrences():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Start with mlflow ui\n")
        f.write("# Also use mlflow ui here\n")
        f.flush()
        temp_file = Path(f.name)

    try:
        result = run_check_mlflow_ui(temp_file)
        assert result.returncode == 1, "Script should fail when 'mlflow ui' is found"
    finally:
        temp_file.unlink()


def test_check_mlflow_ui_case_sensitive():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Run MLFLOW UI to start\n")
        f.flush()
        temp_file = Path(f.name)

    try:
        result = run_check_mlflow_ui(temp_file)
        assert result.returncode == 0, "Script should pass for different casing"
    finally:
        temp_file.unlink()


def test_check_mlflow_ui_clean_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# This is a clean file\n")
        f.write("import mlflow\n")
        f.flush()
        temp_file = Path(f.name)

    try:
        result = run_check_mlflow_ui(temp_file)
        assert result.returncode == 0, "Script should pass on clean files"
    finally:
        temp_file.unlink()
