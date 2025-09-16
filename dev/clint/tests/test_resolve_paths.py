from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from clint.utils import ALLOWED_EXTS, _git_ls_files, _git_ls_untracked_files, resolve_paths


def test_git_ls_files_success() -> None:
    mock_output = "file1.py\ndir/file2.md\nfile3.ipynb\n"

    with patch("subprocess.check_output", return_value=mock_output) as mock_check_output:
        result = _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()
    expected = [Path("file1.py"), Path("dir/file2.md"), Path("file3.ipynb")]
    assert result == expected


def test_git_ls_files_empty_output() -> None:
    with patch("subprocess.check_output", return_value="") as mock_check_output:
        result = _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()
    assert result == []


def test_git_ls_files_with_pathspecs() -> None:
    mock_output = "file1.py\n"

    with patch("subprocess.check_output", return_value=mock_output) as mock_check_output:
        result = _git_ls_files([Path("dir1"), Path("file.py")])

    # Verify the correct git command was called
    mock_check_output.assert_called_once_with(
        ["git", "ls-files", "--", Path("dir1"), Path("file.py")],
        text=True,
    )
    assert result == [Path("file1.py")]


def test_git_ls_files_subprocess_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git-tracked files"):
            _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()


def test_git_ls_files_os_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=OSError("git not found")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git-tracked files"):
            _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()


def test_git_ls_untracked_files_success() -> None:
    mock_output = "untracked1.py\ndir/untracked2.md\nuntracked3.ipynb\n"

    with patch("subprocess.check_output", return_value=mock_output) as mock_check_output:
        result = _git_ls_untracked_files([Path(".")])

    mock_check_output.assert_called_once_with(
        ["git", "ls-files", "--others", "--exclude-standard", "--", Path(".")],
        text=True,
    )
    expected = [Path("untracked1.py"), Path("dir/untracked2.md"), Path("untracked3.ipynb")]
    assert result == expected


def test_git_ls_untracked_files_empty_output() -> None:
    with patch("subprocess.check_output", return_value="") as mock_check_output:
        result = _git_ls_untracked_files([Path(".")])

    mock_check_output.assert_called_once()
    assert result == []


def test_git_ls_untracked_files_subprocess_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git-untracked files"):
            _git_ls_untracked_files([Path(".")])

    mock_check_output.assert_called_once()


def test_git_ls_untracked_files_os_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=OSError("git not found")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git-untracked files"):
            _git_ls_untracked_files([Path(".")])

    mock_check_output.assert_called_once()


def test_resolve_paths_default_current_dir() -> None:
    mock_tracked_output = "file1.py\nfile2.md\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([])

    assert mock_check_output.call_count == 2
    mock_check_output.assert_any_call(["git", "ls-files", "--", Path(".")], text=True)
    mock_check_output.assert_any_call(
        ["git", "ls-files", "--others", "--exclude-standard", "--", Path(".")], text=True
    )
    assert mock_exists.call_count == 2  # Called for each file
    expected = [Path("file1.py"), Path("file2.md")]
    assert result == expected


def test_resolve_paths_filters_by_extension() -> None:
    mock_tracked_output = (
        "file1.py\nfile2.md\nfile3.txt\nfile4.ipynb\nfile5.mdx\nfile6.js\nfile7.rst\n"
    )
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    assert mock_exists.call_count == 5  # Called for each allowed extension file
    expected = [
        Path("file1.py"),
        Path("file2.md"),
        Path("file4.ipynb"),
        Path("file5.mdx"),
        Path("file7.rst"),
    ]
    assert result == expected


def test_resolve_paths_case_insensitive_extensions() -> None:
    mock_tracked_output = "file1.PY\nfile2.MD\nfile3.IPYNB\nfile4.py\nfile5.RST\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    assert mock_exists.call_count == 5  # Called for each file
    expected = [
        Path("file1.PY"),
        Path("file2.MD"),
        Path("file3.IPYNB"),
        Path("file4.py"),
        Path("file5.RST"),
    ]
    assert result == expected


def test_resolve_paths_filters_non_existent_files() -> None:
    mock_tracked_output = "file1.py\nfile2.md\nfile3.py\n"
    mock_untracked_output = ""

    def mock_exists(self: Path) -> bool:
        # Only file1.py and file3.py exist
        return str(self) in ["file1.py", "file3.py"]

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", mock_exists),
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    expected = [Path("file1.py"), Path("file3.py")]
    assert result == expected


def test_resolve_paths_returns_sorted_list() -> None:
    mock_tracked_output = "z_file.py\na_file.md\nm_file.ipynb\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    assert mock_exists.call_count == 3  # Called for each file
    expected = [Path("a_file.md"), Path("m_file.ipynb"), Path("z_file.py")]
    assert result == expected


def test_resolve_paths_deduplicates_results() -> None:
    mock_tracked_output = "file1.py\nfile1.py\nfile2.md\nfile2.md\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    # exists() is called for each file before deduplication
    assert mock_exists.call_count == 4  # Called for all files before deduplication
    expected = [Path("file1.py"), Path("file2.md")]
    assert result == expected


def test_resolve_paths_with_multiple_pathspecs() -> None:
    mock_tracked_output = "dir1/file1.py\ndir2/file2.md\nfile3.ipynb\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path("dir1"), Path("file3.ipynb")])

    # Verify git was called with correct pathspecs for both calls
    assert mock_check_output.call_count == 2
    mock_check_output.assert_any_call(
        ["git", "ls-files", "--", Path("dir1"), Path("file3.ipynb")],
        text=True,
    )
    mock_check_output.assert_any_call(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            Path("dir1"),
            Path("file3.ipynb"),
        ],
        text=True,
    )
    expected = [Path("dir1/file1.py"), Path("dir2/file2.md"), Path("file3.ipynb")]
    assert result == expected


def test_resolve_paths_includes_rst_files() -> None:
    mock_tracked_output = "README.rst\ndocs/index.rst\nsetup.py\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    assert mock_exists.call_count == 3  # Called for each file
    expected = [Path("README.rst"), Path("docs/index.rst"), Path("setup.py")]
    assert result == expected


def test_allowed_extensions_constant() -> None:
    expected = {".md", ".mdx", ".rst", ".py", ".ipynb"}
    assert ALLOWED_EXTS == expected


def test_resolve_paths_includes_untracked_files() -> None:
    mock_tracked_output = "tracked1.py\ntracked2.md\n"
    mock_untracked_output = "untracked1.py\nuntracked2.rst\n"

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    mock_check_output.assert_any_call(["git", "ls-files", "--", Path(".")], text=True)
    mock_check_output.assert_any_call(
        ["git", "ls-files", "--others", "--exclude-standard", "--", Path(".")], text=True
    )
    assert mock_exists.call_count == 4  # Called for each file
    expected = [
        Path("tracked1.py"),
        Path("tracked2.md"),
        Path("untracked1.py"),
        Path("untracked2.rst"),
    ]
    assert result == expected


def test_resolve_paths_deduplicates_tracked_and_untracked() -> None:
    # Test case where same file appears in both tracked and untracked lists
    # (though this shouldn't happen in practice, let's handle it gracefully)
    mock_tracked_output = "file1.py\nfile2.md\n"
    mock_untracked_output = "file1.py\nfile3.rst\n"  # file1.py appears in both

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    # Should deduplicate file1.py
    expected = [Path("file1.py"), Path("file2.md"), Path("file3.rst")]
    assert result == expected


def test_resolve_paths_untracked_only() -> None:
    # Test case where only untracked files exist
    mock_tracked_output = ""
    mock_untracked_output = "untracked1.py\nuntracked2.md\n"

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    expected = [Path("untracked1.py"), Path("untracked2.md")]
    assert result == expected


def test_resolve_paths_tracked_only() -> None:
    # Test case where only tracked files exist (same as before but explicit)
    mock_tracked_output = "tracked1.py\ntracked2.md\n"
    mock_untracked_output = ""

    with (
        patch(
            "subprocess.check_output", side_effect=[mock_tracked_output, mock_untracked_output]
        ) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    assert mock_check_output.call_count == 2
    expected = [Path("tracked1.py"), Path("tracked2.md")]
    assert result == expected
