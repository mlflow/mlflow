import os
import unicodedata

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.paths import (
    assert_regular_tree,
    collect_tree,
    normalize_subpath,
    resolve_contained,
    tree_size,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("skills/review", "skills/review"),
        ("skills//review/", "skills/review"),
        (" a/b ", "a/b"),
    ],
)
def test_normalize_subpath_valid(raw, expected):
    assert normalize_subpath(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["/abs/path", "a/../b", "../x", "./a", "a/./b", "a\\b", "..", "."],
)
def test_normalize_subpath_invalid(raw):
    with pytest.raises(MlflowException, match="Subpath"):
        normalize_subpath(raw)


def test_normalize_subpath_rejects_non_string():
    with pytest.raises(MlflowException, match="must be a string"):
        normalize_subpath(3)


def test_resolve_contained(tmp_path):
    (tmp_path / "skills" / "review").mkdir(parents=True)
    assert resolve_contained(tmp_path, None) == tmp_path
    assert resolve_contained(tmp_path, "skills/review") == tmp_path / "skills" / "review"


def test_resolve_contained_missing_and_file(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(MlflowException, match="does not exist"):
        resolve_contained(tmp_path, "nope")
    with pytest.raises(MlflowException, match="must point to a directory"):
        resolve_contained(tmp_path, "file.txt")


def test_resolve_contained_rejects_symlink_traversal(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MlflowException, match="traverses a symbolic link"):
        resolve_contained(root, "link")


def test_collect_tree_orders_by_path_bytes_and_skips_links(tmp_path):
    (tmp_path / "b.txt").write_bytes(b"bb")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "z.txt").write_bytes(b"z")
    (tmp_path / "0.txt").write_bytes(b"0")
    (tmp_path / "link.txt").symlink_to(tmp_path / "b.txt")
    (tmp_path / "linkdir").symlink_to(tmp_path / "a", target_is_directory=True)

    files = collect_tree(tmp_path)

    # Sorted by UTF-8 byte value: digits precede letters, and "a/" precedes "b".
    assert [f.path for f in files] == ["0.txt", "a/z.txt", "b.txt"]
    assert [f.size for f in files] == [1, 1, 2]
    assert tree_size(tmp_path) == 4


def test_collect_tree_normalizes_nfc_and_rejects_collisions(tmp_path):
    nfd_name = unicodedata.normalize("NFD", "café.txt")
    nfc_name = unicodedata.normalize("NFC", "café.txt")
    (tmp_path / nfd_name).write_bytes(b"1")
    files = collect_tree(tmp_path)
    assert files[0].path == nfc_name

    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / nfd_name).write_bytes(b"1")
    try:
        (tmp_path / "dir" / nfc_name).write_bytes(b"2")
    except FileExistsError:
        pytest.skip("filesystem normalizes Unicode file names itself")
    if len(list((tmp_path / "dir").iterdir())) < 2:
        pytest.skip("filesystem normalizes Unicode file names itself")
    with pytest.raises(MlflowException, match="normalize to the same path"):
        collect_tree(tmp_path)


def test_assert_regular_tree_accepts_plain_tree(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("x")
    assert_regular_tree(tmp_path)


def test_assert_regular_tree_rejects_symlink(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    (tmp_path / "l.txt").symlink_to(tmp_path / "f.txt")
    with pytest.raises(MlflowException, match="symbolic links"):
        assert_regular_tree(tmp_path)


def test_assert_regular_tree_rejects_hard_link(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    os.link(tmp_path / "f.txt", tmp_path / "h.txt")
    with pytest.raises(MlflowException, match="hard links"):
        assert_regular_tree(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not supported on this platform")
def test_assert_regular_tree_rejects_special_file(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(MlflowException, match="only regular files and directories"):
        assert_regular_tree(tmp_path)


def test_assert_regular_tree_rejects_link_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(MlflowException, match="not a link"):
        assert_regular_tree(link)
