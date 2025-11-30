from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from clint.utils import ALLOWED_EXTS, _git_ls_files, resolve_paths


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create and initialize a git repository in a temporary directory."""
    subprocess.check_call(["git", "init"], cwd=tmp_path, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_resolve_paths_with_real_git_repo_tracked_and_untracked(git_repo: Path) -> None:
    tracked_py = git_repo / "tracked.py"
    tracked_md = git_repo / "tracked.md"
    tracked_py.write_text("# tracked python file")
    tracked_md.write_text("# tracked markdown file")

    subprocess.check_call(["git", "add", "tracked.py", "tracked.md"])
    subprocess.check_call(["git", "commit", "-m", "Add tracked files"])

    untracked_py = git_repo / "untracked.py"
    untracked_rst = git_repo / "untracked.rst"
    untracked_py.write_text("# untracked python file")
    untracked_rst.write_text("# untracked rst file")

    gitignore = git_repo / ".gitignore"
    gitignore.write_text("ignored.py\n")
    ignored_py = git_repo / "ignored.py"
    ignored_py.write_text("# ignored file")

    result = resolve_paths([Path(".")])

    expected_paths = [
        Path("tracked.md"),
        Path("tracked.py"),
        Path("untracked.py"),
        Path("untracked.rst"),
    ]
    assert result == expected_paths
    assert Path("ignored.py") not in result


def test_resolve_paths_with_real_git_repo_specific_pathspecs(git_repo: Path) -> None:
    subdir = git_repo / "subdir"
    subdir.mkdir()

    root_py = git_repo / "root.py"
    subdir_py = subdir / "sub.py"
    subdir_md = subdir / "sub.md"

    root_py.write_text("# root file")
    subdir_py.write_text("# subdir python file")
    subdir_md.write_text("# subdir markdown file")

    subprocess.check_call(["git", "add", "root.py"])
    subprocess.check_call(["git", "commit", "-m", "Add root file"])

    result = resolve_paths([Path("subdir")])

    expected_paths = [Path("subdir/sub.md"), Path("subdir/sub.py")]
    assert result == expected_paths
    assert Path("root.py") not in result


def test_resolve_paths_with_real_git_repo_untracked_only(git_repo: Path) -> None:
    untracked1 = git_repo / "untracked1.py"
    untracked2 = git_repo / "untracked2.md"
    untracked1.write_text("# untracked file 1")
    untracked2.write_text("# untracked file 2")

    result = resolve_paths([Path(".")])

    expected_paths = [Path("untracked1.py"), Path("untracked2.md")]
    assert result == expected_paths


def test_resolve_paths_with_real_git_repo_tracked_only(git_repo: Path) -> None:
    tracked1 = git_repo / "tracked1.py"
    tracked2 = git_repo / "tracked2.md"
    tracked1.write_text("# tracked file 1")
    tracked2.write_text("# tracked file 2")

    subprocess.check_call(["git", "add", "tracked1.py", "tracked2.md"])
    subprocess.check_call(["git", "commit", "-m", "Add tracked files"])

    result = resolve_paths([Path(".")])

    expected_paths = [Path("tracked1.py"), Path("tracked2.md")]
    assert result == expected_paths


def test_resolve_paths_with_real_git_repo_removed_tracked_file(git_repo: Path) -> None:
    tracked1 = git_repo / "tracked1.py"
    tracked2 = git_repo / "tracked2.md"
    tracked1.write_text("# tracked file 1")
    tracked2.write_text("# tracked file 2")

    subprocess.check_call(["git", "add", "tracked1.py", "tracked2.md"])
    subprocess.check_call(["git", "commit", "-m", "Add tracked files"])

    tracked1.unlink()

    result = resolve_paths([Path(".")])

    expected_paths = [Path("tracked2.md")]
    assert result == expected_paths


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

    mock_check_output.assert_called_once()
    assert result == [Path("file1.py")]


def test_git_ls_files_subprocess_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git files"):
            _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()


def test_git_ls_files_os_error() -> None:
    with patch(
        "subprocess.check_output", side_effect=OSError("git not found")
    ) as mock_check_output:
        with pytest.raises(RuntimeError, match="Failed to list git files"):
            _git_ls_files([Path(".")])

    mock_check_output.assert_called_once()


def test_resolve_paths_default_current_dir() -> None:
    mock_output = "file1.py\nfile2.md\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([])

    mock_check_output.assert_called_once()
    expected = [Path("file1.py"), Path("file2.md")]
    assert result == expected


def test_resolve_paths_filters_by_extension() -> None:
    mock_output = "file1.py\nfile2.md\nfile3.txt\nfile4.ipynb\nfile5.mdx\nfile6.js\nfile7.rst\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    mock_check_output.assert_called_once()
    expected = [
        Path("file1.py"),
        Path("file2.md"),
        Path("file4.ipynb"),
        Path("file5.mdx"),
        Path("file7.rst"),
    ]
    assert result == expected


def test_resolve_paths_case_insensitive_extensions() -> None:
    mock_output = "file1.PY\nfile2.MD\nfile3.IPYNB\nfile4.py\nfile5.RST\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    mock_check_output.assert_called_once()
    expected = [
        Path("file1.PY"),
        Path("file2.MD"),
        Path("file3.IPYNB"),
        Path("file4.py"),
        Path("file5.RST"),
    ]
    assert result == expected


def test_resolve_paths_returns_sorted_list() -> None:
    mock_output = "z_file.py\na_file.md\nm_file.ipynb\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    mock_check_output.assert_called_once()
    expected = [Path("a_file.md"), Path("m_file.ipynb"), Path("z_file.py")]
    assert result == expected


def test_resolve_paths_deduplicates_results() -> None:
    mock_output = "file1.py\nfile1.py\nfile2.md\nfile2.md\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    mock_check_output.assert_called_once()
    expected = [Path("file1.py"), Path("file2.md")]
    assert result == expected


def test_resolve_paths_with_multiple_pathspecs() -> None:
    mock_output = "dir1/file1.py\ndir2/file2.md\nfile3.ipynb\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path("dir1"), Path("file3.ipynb")])

    mock_check_output.assert_called_once()
    expected = [Path("dir1/file1.py"), Path("dir2/file2.md"), Path("file3.ipynb")]
    assert result == expected


def test_resolve_paths_includes_rst_files() -> None:
    mock_output = "README.rst\ndocs/index.rst\nsetup.py\n"

    with (
        patch("subprocess.check_output", return_value=mock_output) as mock_check_output,
        patch("pathlib.Path.exists", return_value=True),
    ):
        result = resolve_paths([Path(".")])

    mock_check_output.assert_called_once()
    expected = [Path("README.rst"), Path("docs/index.rst"), Path("setup.py")]
    assert result == expected


def test_allowed_extensions_constant() -> None:
    expected = {".md", ".mdx", ".rst", ".py", ".ipynb"}
    assert ALLOWED_EXTS == expected
