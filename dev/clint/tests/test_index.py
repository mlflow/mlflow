"""Tests for SymbolIndex functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from clint.index import SymbolIndex


def test_symbol_index_build_skips_missing_files():
    """Test that SymbolIndex.build() skips files listed by git but don't exist on filesystem."""

    # Create a temporary directory to simulate a git repo
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        mlflow_dir = repo_root / "mlflow"
        mlflow_dir.mkdir()

        # Create one existing file
        existing_file = mlflow_dir / "existing.py"
        existing_file.write_text("def existing_function(): pass")

        # Mock git ls-files to return both existing and non-existing files
        mock_git_output = "mlflow/existing.py\nmlflow/deleted.py\n"

        with (
            patch("clint.index.get_repo_root", return_value=repo_root),
            patch("subprocess.check_output", return_value=mock_git_output),
        ):
            # This should now work without raising an exception
            # The fix should skip missing files gracefully
            index = SymbolIndex.build()
            assert isinstance(index, SymbolIndex)
            # The index should be created successfully, even with missing files


def test_symbol_index_build_basic():
    """Test that SymbolIndex.build() works with existing files."""

    # Create a temporary directory to simulate a git repo
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        mlflow_dir = repo_root / "mlflow"
        mlflow_dir.mkdir()

        # Create a simple test file
        test_file = mlflow_dir / "test.py"
        test_file.write_text("def test_function(): pass")

        # Mock git ls-files to return only the existing file
        mock_git_output = "mlflow/test.py\n"

        with (
            patch("clint.index.get_repo_root", return_value=repo_root),
            patch("subprocess.check_output", return_value=mock_git_output),
        ):
            # This should work fine
            index = SymbolIndex.build()
            assert isinstance(index, SymbolIndex)
