import gzip
import io
import stat
import tarfile
import zipfile

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import (
    MAX_ARCHIVE_ENTRIES,
    extract_skill_archive,
    extract_zip_archive,
    get_max_decompressed_size,
    package_skill_tree,
    validate_skill_archive,
    validate_zip_archive,
)
from mlflow.genai.skill_content.digest import compute_tree_digest


def _write_tree(root, files):
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _make_tar(path, members, *, compressed=True):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, kind, content in members:
            info = tarfile.TarInfo(name=name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = content.decode()
                tar.addfile(info)
    data = buffer.getvalue()
    path.write_bytes(gzip.compress(data) if compressed else data)
    return path


def test_get_max_decompressed_size(monkeypatch):
    assert get_max_decompressed_size() == 25 * 1024**2
    monkeypatch.setenv("MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE", "123")
    assert get_max_decompressed_size() == 123
    assert get_max_decompressed_size(7) == 7
    with pytest.raises(MlflowException, match="must be positive"):
        get_max_decompressed_size(0)


def test_package_roundtrip_preserves_digest_and_is_reproducible(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _write_tree(source, {"SKILL.md": b"---\nname: demo\n---\n", "ref/data.txt": b"1234"})
    (source / "link.txt").symlink_to(source / "SKILL.md")

    first = package_skill_tree(source, tmp_path / "a.tar.gz")
    (source / "SKILL.md").touch()
    second = package_skill_tree(source, tmp_path / "b.tar.gz")
    assert first.read_bytes() == second.read_bytes()

    extracted = extract_skill_archive(first, tmp_path / "out")
    assert sorted(p.name for p in extracted.rglob("*") if p.is_file()) == ["SKILL.md", "data.txt"]
    assert not (extracted / "link.txt").exists()
    assert compute_tree_digest(extracted) == compute_tree_digest(source)
    assert validate_skill_archive(first) == len(b"---\nname: demo\n---\n") + 4


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([("../escape.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("/abs.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("dir\\file.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("a/./b.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("link", tarfile.SYMTYPE, b"SKILL.md")], "not a regular file or directory"),
        ([("hard", tarfile.LNKTYPE, b"SKILL.md")], "not a regular file or directory"),
        ([("fifo", tarfile.FIFOTYPE, b"")], "not a regular file or directory"),
        ([("dev", tarfile.CHRTYPE, b"")], "not a regular file or directory"),
        ([(".", tarfile.REGTYPE, b"x")], "root entry must be a directory"),
    ],
)
def test_validate_skill_archive_rejects_unsafe_members(tmp_path, members, message):
    archive = _make_tar(tmp_path / "bad.tar.gz", members)
    with pytest.raises(MlflowException, match=message):
        validate_skill_archive(archive)
    assert not (tmp_path / "out").exists()


def test_validate_skill_archive_accepts_dot_prefixed_and_dir_entries(tmp_path):
    archive = _make_tar(
        tmp_path / "ok.tar.gz",
        [
            ("./", tarfile.DIRTYPE, b""),
            ("./skills/", tarfile.DIRTYPE, b""),
            ("./skills/SKILL.md", tarfile.REGTYPE, b"abc"),
        ],
    )
    assert validate_skill_archive(archive) == 3
    out = extract_skill_archive(archive, tmp_path / "out")
    assert (out / "skills" / "SKILL.md").read_bytes() == b"abc"


def test_validate_skill_archive_enforces_size_limit(tmp_path):
    archive = _make_tar(tmp_path / "big.tar.gz", [("a.bin", tarfile.REGTYPE, b"x" * 100)])
    assert validate_skill_archive(archive, max_bytes=100) == 100
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_skill_archive(archive, max_bytes=99)
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        extract_skill_archive(archive, tmp_path / "out", max_bytes=99)


def test_validate_skill_archive_size_limit_from_env(tmp_path, monkeypatch):
    archive = _make_tar(tmp_path / "big.tar.gz", [("a.bin", tarfile.REGTYPE, b"x" * 100)])
    monkeypatch.setenv("MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE", "50")
    with pytest.raises(MlflowException, match="limit of 50 bytes"):
        validate_skill_archive(archive)


def test_validate_skill_archive_enforces_entry_count(tmp_path):
    members = [(f"f{i}", tarfile.REGTYPE, b"") for i in range(MAX_ARCHIVE_ENTRIES + 1)]
    archive = _make_tar(tmp_path / "many.tar.gz", members)
    with pytest.raises(MlflowException, match="more than"):
        validate_skill_archive(archive)


def test_validate_skill_archive_rejects_non_archive(tmp_path):
    bogus = tmp_path / "bogus.tar.gz"
    bogus.write_bytes(b"not an archive")
    with pytest.raises(MlflowException, match="not a readable tar archive"):
        validate_skill_archive(bogus)


def test_extract_skill_archive_uncompressed(tmp_path):
    archive = _make_tar(
        tmp_path / "plain.tar", [("SKILL.md", tarfile.REGTYPE, b"x")], compressed=False
    )
    out = extract_skill_archive(archive, tmp_path / "out", compressed=False)
    assert (out / "SKILL.md").read_bytes() == b"x"


def test_extract_skill_archive_requires_empty_destination(tmp_path):
    archive = _make_tar(tmp_path / "ok.tar.gz", [("SKILL.md", tarfile.REGTYPE, b"x")])
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "existing").write_text("x")
    with pytest.raises(MlflowException, match="must be an empty directory"):
        extract_skill_archive(archive, dest)


def test_extract_skill_archive_enforces_actual_bytes_not_headers(tmp_path):
    # A header can under-declare its size; the budget must track bytes really written.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name="a.bin")
        info.size = 10
        tar.addfile(info, io.BytesIO(b"x" * 10))
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(gzip.compress(buffer.getvalue()))
    out = extract_skill_archive(archive, tmp_path / "out", max_bytes=10)
    assert (out / "a.bin").stat().st_size == 10


def _make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, content)
    return path


def test_zip_roundtrip(tmp_path):
    archive = _make_zip(
        tmp_path / "ok.zip",
        [
            ("skill/", b"", None),
            ("skill/SKILL.md", b"abc", None),
            ("skill/a/b.txt", b"12", 0o100644),
        ],
    )
    assert validate_zip_archive(archive) == 5
    out = extract_zip_archive(archive, tmp_path / "out")
    assert (out / "skill" / "SKILL.md").read_bytes() == b"abc"
    assert (out / "skill" / "a" / "b.txt").read_bytes() == b"12"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([("../x.txt", b"x", None)], "unsafe path"),
        ([("/abs.txt", b"x", None)], "unsafe path"),
        ([("a\\b.txt", b"x", None)], "unsafe path"),
        ([("link", b"SKILL.md", stat.S_IFLNK | 0o777)], "not a regular file or directory"),
        ([("fifo", b"", stat.S_IFIFO | 0o644)], "not a regular file or directory"),
    ],
)
def test_validate_zip_archive_rejects_unsafe_entries(tmp_path, entries, message):
    archive = _make_zip(tmp_path / "bad.zip", entries)
    with pytest.raises(MlflowException, match=message):
        validate_zip_archive(archive)


def test_validate_zip_archive_size_limit_and_bad_file(tmp_path):
    archive = _make_zip(tmp_path / "big.zip", [("a.bin", b"x" * 100, None)])
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_zip_archive(archive, max_bytes=99)
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"nope")
    with pytest.raises(MlflowException, match="not a readable ZIP archive"):
        validate_zip_archive(bogus)
