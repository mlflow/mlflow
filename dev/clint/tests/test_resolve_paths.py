from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from clint.utils import ALLOWED_EXTS, _git_ls_files, resolve_paths


class TestGitLsFiles:
    def test_git_ls_files_success(self):
        """Test that _git_ls_files returns correct paths when git command succeeds."""
        mock_output = "file1.py\ndir/file2.md\nfile3.ipynb\n"

        with patch("subprocess.check_output", return_value=mock_output):
            result = _git_ls_files([Path(".")])

        expected = [Path("file1.py"), Path("dir/file2.md"), Path("file3.ipynb")]
        assert result == expected

    def test_git_ls_files_empty_output(self):
        """Test that _git_ls_files handles empty git output."""
        with patch("subprocess.check_output", return_value=""):
            result = _git_ls_files([Path(".")])

        assert result == []

    def test_git_ls_files_with_pathspecs(self):
        """Test that _git_ls_files passes pathspecs correctly to git."""
        mock_output = "file1.py\n"

        with patch("subprocess.check_output", return_value=mock_output) as mock_check_output:
            result = _git_ls_files([Path("dir1"), Path("file.py")])

        # Verify the correct git command was called
        mock_check_output.assert_called_once_with(
            ["git", "ls-files", "--", Path("dir1"), Path("file.py")],
            text=True,
        )
        assert result == [Path("file1.py")]

    def test_git_ls_files_subprocess_error(self):
        """Test that _git_ls_files raises RuntimeError on subprocess error."""
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            with pytest.raises(RuntimeError, match="Failed to list git-tracked files"):
                _git_ls_files([Path(".")])

    def test_git_ls_files_os_error(self):
        """Test that _git_ls_files raises RuntimeError on OS error."""
        with patch("subprocess.check_output", side_effect=OSError("git not found")):
            with pytest.raises(RuntimeError, match="Failed to list git-tracked files"):
                _git_ls_files([Path(".")])


class TestResolvePaths:
    def test_resolve_paths_default_current_dir(self):
        """Test that resolve_paths defaults to current directory when no paths provided."""
        mock_git_output = "file1.py\nfile2.md\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([])

        expected = [Path("file1.py"), Path("file2.md")]
        assert result == expected

    def test_resolve_paths_filters_by_extension(self):
        """Test that resolve_paths only includes allowed extensions."""
        mock_git_output = "file1.py\nfile2.md\nfile3.txt\nfile4.ipynb\nfile5.mdx\nfile6.js\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([Path(".")])

        expected = [Path("file1.py"), Path("file2.md"), Path("file4.ipynb"), Path("file5.mdx")]
        assert result == expected

    def test_resolve_paths_case_insensitive_extensions(self):
        """Test that resolve_paths handles case-insensitive extensions."""
        mock_git_output = "file1.PY\nfile2.MD\nfile3.IPYNB\nfile4.py\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([Path(".")])

        expected = [Path("file1.PY"), Path("file2.MD"), Path("file3.IPYNB"), Path("file4.py")]
        assert result == expected

    def test_resolve_paths_filters_non_existent_files(self):
        """Test that resolve_paths excludes files that don't exist in working tree."""
        mock_git_output = "file1.py\nfile2.md\nfile3.py\n"

        def mock_exists(self):
            # Only file1.py and file3.py exist
            return str(self) in ["file1.py", "file3.py"]

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", mock_exists),
        ):
            result = resolve_paths([Path(".")])

        expected = [Path("file1.py"), Path("file3.py")]
        assert result == expected

    def test_resolve_paths_returns_sorted_list(self):
        """Test that resolve_paths returns a sorted list."""
        mock_git_output = "z_file.py\na_file.md\nm_file.ipynb\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([Path(".")])

        expected = [Path("a_file.md"), Path("m_file.ipynb"), Path("z_file.py")]
        assert result == expected

    def test_resolve_paths_deduplicates_results(self):
        """Test that resolve_paths removes duplicates."""
        mock_git_output = "file1.py\nfile1.py\nfile2.md\nfile2.md\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([Path(".")])

        expected = [Path("file1.py"), Path("file2.md")]
        assert result == expected

    def test_resolve_paths_with_multiple_pathspecs(self):
        """Test that resolve_paths works with multiple input paths."""
        mock_git_output = "dir1/file1.py\ndir2/file2.md\nfile3.ipynb\n"

        with (
            patch("subprocess.check_output", return_value=mock_git_output) as mock_check_output,
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = resolve_paths([Path("dir1"), Path("file3.ipynb")])

        # Verify git was called with correct pathspecs
        mock_check_output.assert_called_once_with(
            ["git", "ls-files", "--", Path("dir1"), Path("file3.ipynb")],
            text=True,
        )
        expected = [Path("dir1/file1.py"), Path("dir2/file2.md"), Path("file3.ipynb")]
        assert result == expected


class TestAllowedExts:
    def test_allowed_extensions_constant(self):
        """Test that ALLOWED_EXTS contains the expected extensions."""
        expected = {".md", ".mdx", ".py", ".ipynb"}
        assert ALLOWED_EXTS == expected
