from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from clint.index import SymbolIndex, extract_symbols_from_file
from clint.utils import get_repo_root


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create and initialize a git repository in a temporary directory."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    monkeypatch.chdir(repo_path)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)

    # Create mlflow directory and some test files
    mlflow_dir = repo_path / "mlflow"
    mlflow_dir.mkdir()

    # Create a simple Python file
    test_file = mlflow_dir / "test_module.py"
    test_file.write_text("""
def test_function(arg1, arg2="default"):
    '''Test function'''
    return arg1 + arg2

class TestClass:
    def __init__(self, name):
        self.name = name

    def method(self):
        return self.name
""")

    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True
    )

    return repo_path


def test_get_repo_root_caching() -> None:
    """Test that get_repo_root caches results."""
    # Clear cache first
    get_repo_root.cache_clear()

    with patch("subprocess.check_output", return_value="/test/repo\n") as mock_subprocess:
        # First call
        root1 = get_repo_root()
        # Second call
        root2 = get_repo_root()

        # Should only call subprocess once due to caching
        assert mock_subprocess.call_count == 1
        assert root1 == root2 == Path("/test/repo")


def test_get_repo_root_error() -> None:
    """Test that get_repo_root raises RuntimeError on git command failure."""
    # Clear cache first
    get_repo_root.cache_clear()

    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
        with pytest.raises(RuntimeError, match="Failed to find git repository root"):
            get_repo_root()


def test_extract_symbols_from_file() -> None:
    """Test extracting symbols from a Python file."""
    content = '''
def test_function(arg1, arg2="default"):
    """Test function"""
    return arg1 + arg2

class TestClass:
    def __init__(self, name):
        self.name = name
'''

    result = extract_symbols_from_file("mlflow/test_module.py", content)
    assert result is not None

    import_mapping, func_mapping = result

    # Should have function and class
    assert "mlflow.test_module.test_function" in func_mapping
    assert "mlflow.test_module.TestClass" in func_mapping

    # Check function info
    func_info = func_mapping["mlflow.test_module.test_function"]
    assert func_info.args == ["arg1", "arg2"]
    assert not func_info.has_vararg
    assert not func_info.has_kwarg


def test_extract_symbols_from_file_non_mlflow() -> None:
    """Test that non-mlflow files are skipped."""
    content = "def test_function(): pass"
    result = extract_symbols_from_file("other/module.py", content)
    assert result is None


def test_symbol_index_build_works_from_subdirectory(git_repo: Path) -> None:
    """Test that SymbolIndex.build() works correctly from any subdirectory."""
    # Test from repository root
    original_cwd = Path.cwd()
    try:
        # Change to repo root and test
        os.chdir(git_repo)
        get_repo_root.cache_clear()  # Clear cache to ensure fresh call
        index_from_root = SymbolIndex.build()

        # Change to subdirectory and test
        subdir = git_repo / "subdir"
        subdir.mkdir()
        os.chdir(subdir)
        get_repo_root.cache_clear()  # Clear cache to ensure fresh call
        index_from_subdir = SymbolIndex.build()

        # Should get same results regardless of working directory
        assert len(index_from_root.func_mapping) == len(index_from_subdir.func_mapping)
        assert len(index_from_root.import_mapping) == len(index_from_subdir.import_mapping)

        # Should contain our test function
        assert "mlflow.test_module.test_function" in index_from_root.func_mapping
        assert "mlflow.test_module.test_function" in index_from_subdir.func_mapping

    finally:
        os.chdir(original_cwd)


def test_symbol_index_build_min_workers() -> None:
    """Test that SymbolIndex.build() handles minimum workers correctly."""
    with (
        patch("clint.index.get_repo_root", return_value=Path("/test/repo")),
        patch("subprocess.check_output", return_value=""),  # No files found
        patch("clint.index.ProcessPoolExecutor") as mock_executor,
    ):
        SymbolIndex.build()

        # Should use at least 1 worker even when no files are found
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["max_workers"] >= 1


def test_symbol_index_build_handles_file_read_errors() -> None:
    """Test that SymbolIndex.build() handles file read errors gracefully."""
    mock_repo_root = Path("/test/repo")

    with (
        patch("clint.index.get_repo_root", return_value=mock_repo_root),
        patch("subprocess.check_output", return_value="mlflow/test.py\n"),
        patch.object(Path, "read_text", side_effect=OSError("File not found")),
    ):
        # Should not raise exception even if file can't be read
        index = SymbolIndex.build()
        assert isinstance(index, SymbolIndex)
