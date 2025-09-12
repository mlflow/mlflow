#!/usr/bin/env python3
"""
Simple functional test for dev/pyproject.py optimization.

This test verifies that the pyproject.py script only writes files when content changes.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the MLflow root to the Python path
test_dir = Path(__file__).parent
dev_dir = test_dir.parent
mlflow_root = dev_dir.parent
sys.path.insert(0, str(mlflow_root))

# Import the pyproject module directly
import importlib.util

spec = importlib.util.spec_from_file_location("pyproject", dev_dir / "pyproject.py")
pyproject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyproject)

write_file_if_changed = pyproject.write_file_if_changed
write_toml_file_if_changed = pyproject.write_toml_file_if_changed


def test_write_file_if_changed():
    """Test the write_file_if_changed function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        content = "Hello, world!"

        # Test writing new file
        result = write_file_if_changed(file_path, content)
        assert result is True, "Should write new file"
        assert file_path.read_text() == content, "Content should match"

        # Test skipping write when content is the same
        original_mtime = file_path.stat().st_mtime
        result = write_file_if_changed(file_path, content)
        assert result is False, "Should skip write when content unchanged"
        assert file_path.stat().st_mtime == original_mtime, "Mtime should not change"

        # Test writing when content changes
        new_content = "New content"
        result = write_file_if_changed(file_path, new_content)
        assert result is True, "Should write when content changes"
        assert file_path.read_text() == new_content, "Content should be updated"

    print("✓ write_file_if_changed tests passed")


def test_write_toml_file_if_changed():
    """Test the write_toml_file_if_changed function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.toml"
        description = "# Test file\n"
        toml_data = {"project": {"name": "test", "version": "1.0.0"}}

        # Test writing new TOML file
        result = write_toml_file_if_changed(file_path, description, toml_data)
        assert result is True, "Should write new TOML file"
        assert file_path.exists(), "File should exist"

        # Test skipping write when content is the same
        original_mtime = file_path.stat().st_mtime
        result = write_toml_file_if_changed(file_path, description, toml_data)
        assert result is False, "Should skip write when TOML content unchanged"
        assert file_path.stat().st_mtime == original_mtime, "Mtime should not change"

        # Test writing when content changes
        toml_data["project"]["version"] = "2.0.0"
        result = write_toml_file_if_changed(file_path, description, toml_data)
        assert result is True, "Should write when TOML content changes"

    print("✓ write_toml_file_if_changed tests passed")


def test_pyproject_script_optimization():
    """Test that running the pyproject.py script doesn't unnecessarily update files."""
    import subprocess

    # Save current working directory
    original_cwd = os.getcwd()

    try:
        # Change to MLflow root directory
        mlflow_root = Path(__file__).parent.parent.parent
        os.chdir(mlflow_root)

        # Get current timestamps of pyproject files
        files_to_check = [
            "pyproject.toml",
            "libs/skinny/pyproject.toml",
            "libs/tracing/pyproject.toml",
            "pyproject.release.toml",
        ]

        original_mtimes = {}
        for file_name in files_to_check:
            file_path = Path(file_name)
            if file_path.exists():
                original_mtimes[file_name] = file_path.stat().st_mtime

        # Add bin to PATH for taplo
        env = os.environ.copy()
        env["PATH"] = f"{mlflow_root}/bin:{env['PATH']}"

        # Run the pyproject.py script
        result = subprocess.run(
            [sys.executable, "dev/pyproject.py"], capture_output=True, text=True, env=env
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check that file timestamps haven't changed (allowing small timing differences)
        for file_name in files_to_check:
            if file_name in original_mtimes:
                file_path = Path(file_name)
                current_mtime = file_path.stat().st_mtime
                time_diff = abs(current_mtime - original_mtimes[file_name])
                assert time_diff < 1.0, f"{file_name} was unnecessarily modified"

    finally:
        os.chdir(original_cwd)

    print("✓ pyproject.py script optimization test passed")


if __name__ == "__main__":
    test_write_file_if_changed()
    test_write_toml_file_if_changed()
    test_pyproject_script_optimization()
    print("\n🎉 All tests passed! The pyproject.py optimization is working correctly.")
