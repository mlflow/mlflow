import hashlib
import struct
import unicodedata

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest


def _write_tree(root, files):
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _expected_digest(files):
    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.encode("utf-8")):
        path_bytes = path.encode("utf-8")
        content = files[path]
        hasher.update(struct.pack(">Q", len(path_bytes)))
        hasher.update(path_bytes)
        hasher.update(struct.pack(">Q", len(content)))
        hasher.update(content)
    return hasher.hexdigest()


def test_digest_matches_canonical_serialization(tmp_path):
    files = {"SKILL.md": b"---\nname: demo\n---\nbody", "scripts/run.py": b"print(1)\n"}
    _write_tree(tmp_path, files)
    digest = compute_tree_digest(tmp_path)
    assert digest == _expected_digest(files)
    assert len(digest) == 64
    assert digest == digest.lower()


def test_digest_is_independent_of_creation_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_tree(first, {"z.txt": b"z", "a/b.txt": b"ab"})
    _write_tree(second, {"a/b.txt": b"ab", "z.txt": b"z"})
    assert compute_tree_digest(first) == compute_tree_digest(second)


def test_digest_framing_prevents_boundary_collisions(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_tree(first, {"a": b"bc"})
    _write_tree(second, {"ab": b"c"})
    assert compute_tree_digest(first) != compute_tree_digest(second)


def test_digest_uses_nfc_paths(tmp_path):
    nfd = tmp_path / "nfd"
    nfc = tmp_path / "nfc"
    nfd.mkdir()
    nfc.mkdir()
    _write_tree(nfd, {unicodedata.normalize("NFD", "café.md"): b"x"})
    _write_tree(nfc, {unicodedata.normalize("NFC", "café.md"): b"x"})
    assert compute_tree_digest(nfd) == compute_tree_digest(nfc)


def test_digest_excludes_symlinks(tmp_path):
    _write_tree(tmp_path, {"SKILL.md": b"x"})
    baseline = compute_tree_digest(tmp_path)
    (tmp_path / "link.md").symlink_to(tmp_path / "SKILL.md")
    assert compute_tree_digest(tmp_path) == baseline


def test_digest_empty_tree(tmp_path):
    assert compute_tree_digest(tmp_path) == hashlib.sha256().hexdigest()


def test_digest_rejects_missing_root(tmp_path):
    with pytest.raises(MlflowException, match="not a directory"):
        compute_tree_digest(tmp_path / "missing")
