import gzip
import io
import stat
import tarfile
import unicodedata
import warnings
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

_NFD = unicodedata.normalize("NFD", "café.md")
_NFC = unicodedata.normalize("NFC", "café.md")


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
            if kind in (tarfile.REGTYPE, tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME):
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


def test_package_roundtrip_keeps_exact_names_with_whitespace(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _write_tree(source, {"a.md ": b"1", " lead.md": b"2", "d/ x ": b"3"})
    archive = package_skill_tree(source, tmp_path / "ws.tar.gz")
    extracted = extract_skill_archive(archive, tmp_path / "out")
    assert sorted(
        p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file()
    ) == [
        " lead.md",
        "a.md ",
        "d/ x ",
    ]
    assert compute_tree_digest(extracted) == compute_tree_digest(source)


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([("../escape.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("/abs.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("dir\\file.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("a/./b.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("C:evil.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("C:/evil.txt", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("C:", tarfile.DIRTYPE, b"")], "unsafe path"),
        ([("CON", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("docs/nul.md", tarfile.REGTYPE, b"x")], "unsafe path"),
        ([("link", tarfile.SYMTYPE, b"SKILL.md")], "not a regular file or directory"),
        ([("hard", tarfile.LNKTYPE, b"SKILL.md")], "not a regular file or directory"),
        ([("fifo", tarfile.FIFOTYPE, b"")], "not a regular file or directory"),
        ([("dev", tarfile.CHRTYPE, b"")], "not a regular file or directory"),
        ([(".", tarfile.REGTYPE, b"x")], "root entry must be a directory"),
        ([("a.md", tarfile.REGTYPE, b"1"), ("a.md", tarfile.REGTYPE, b"2")], "more than once"),
        (
            [(_NFD, tarfile.REGTYPE, b"1"), (_NFC, tarfile.REGTYPE, b"2")],
            "normalize to the same path",
        ),
        (
            [("A.md", tarfile.REGTYPE, b"1"), ("a.md", tarfile.REGTYPE, b"2")],
            "differ only by letter case",
        ),
        (
            [("Dir/x", tarfile.REGTYPE, b"1"), ("dir/y", tarfile.REGTYPE, b"2")],
            "differ only by letter case",
        ),
        (
            [("x", tarfile.REGTYPE, b"1"), ("x/y", tarfile.REGTYPE, b"2")],
            "as a directory, but it is a file",
        ),
        ([("x/y", tarfile.REGTYPE, b"1"), ("x", tarfile.REGTYPE, b"2")], "is also a directory"),
        (
            [("x", tarfile.REGTYPE, b"1"), ("x/", tarfile.DIRTYPE, b"")],
            "as a directory, but it is a file",
        ),
    ],
)
def test_validate_skill_archive_rejects_unsafe_members(tmp_path, members, message):
    archive = _make_tar(tmp_path / "bad.tar.gz", members)
    with pytest.raises(MlflowException, match=message):
        validate_skill_archive(archive)
    with pytest.raises(MlflowException, match=message):
        extract_skill_archive(archive, tmp_path / "out")
    assert not any((tmp_path / "out").rglob("*"))


def test_validate_skill_archive_accepts_dot_prefixed_and_dir_entries(tmp_path):
    archive = _make_tar(
        tmp_path / "ok.tar.gz",
        [
            ("./", tarfile.DIRTYPE, b""),
            ("./skills/", tarfile.DIRTYPE, b""),
            ("./skills/SKILL.md", tarfile.REGTYPE, b"abc"),
            ("skills/", tarfile.DIRTYPE, b""),
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


@pytest.mark.parametrize("header_type", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_validate_skill_archive_bounds_metadata_headers(tmp_path, header_type):
    # A PAX or GNU long-name header declares its payload size and the tar parser reads the
    # whole payload before surfacing any member; the bounded stream must stop it.
    payload = b"\0" * (32 * 1024 * 1024)
    archive = _make_tar(
        tmp_path / "bomb.tar.gz",
        [("hdr", header_type, payload), ("SKILL.md", tarfile.REGTYPE, b"x")],
    )
    assert archive.stat().st_size < 1024 * 1024
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_skill_archive(archive, max_bytes=1024)


def test_validate_skill_archive_enforces_entry_count(tmp_path):
    members = [(f"f{i}", tarfile.REGTYPE, b"") for i in range(MAX_ARCHIVE_ENTRIES + 1)]
    archive = _make_tar(tmp_path / "many.tar.gz", members)
    with pytest.raises(MlflowException, match="more than"):
        validate_skill_archive(archive)


def test_validate_skill_archive_rejects_non_archive_and_corrupt_gzip(tmp_path):
    bogus = tmp_path / "bogus.tar.gz"
    bogus.write_bytes(b"not an archive")
    with pytest.raises(MlflowException, match="(?i)tar archive|malformed"):
        validate_skill_archive(bogus)

    good = _make_tar(tmp_path / "good.tar.gz", [("a.bin", tarfile.REGTYPE, b"x" * 5000)])
    data = bytearray(good.read_bytes())
    data[len(data) // 2 : len(data) // 2 + 8] = b"\xff" * 8
    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(bytes(data))
    with pytest.raises(MlflowException, match="(?i)tar archive|malformed"):
        validate_skill_archive(corrupt)


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
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name="a.bin")
        info.size = 10
        tar.addfile(info, io.BytesIO(b"x" * 10))
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(gzip.compress(buffer.getvalue()))
    out = extract_skill_archive(archive, tmp_path / "out", max_bytes=10)
    assert (out / "a.bin").stat().st_size == 10


def test_skill_archive_subpath_limits_extraction_and_budget(tmp_path):
    archive = _make_tar(
        tmp_path / "pkg.tar.gz",
        [
            ("skills/a/SKILL.md", tarfile.REGTYPE, b"abc"),
            ("other/big.bin", tarfile.REGTYPE, b"x" * 100),
            ("../evil", tarfile.REGTYPE, b"x"),
        ],
    )
    # Entries outside the subpath are still validated for safety.
    with pytest.raises(MlflowException, match="unsafe path"):
        validate_skill_archive(archive, max_bytes=10, subpath="skills/a")

    archive = _make_tar(
        tmp_path / "pkg2.tar.gz",
        [
            ("skills/a/SKILL.md", tarfile.REGTYPE, b"abc"),
            ("other/big.bin", tarfile.REGTYPE, b"x" * 100),
        ],
    )
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_skill_archive(archive, max_bytes=10)
    assert validate_skill_archive(archive, max_bytes=10, subpath="skills/a") == 3
    out = extract_skill_archive(archive, tmp_path / "out", max_bytes=10, subpath="skills/a")
    assert (out / "skills" / "a" / "SKILL.md").read_bytes() == b"abc"
    assert not (out / "other").exists()


def _make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf, warnings.catch_warnings():
        # Deliberately hostile archives include duplicate names, which zipfile warns about.
        warnings.simplefilter("ignore", UserWarning)
        for name, content, attr in entries:
            info = zipfile.ZipInfo(name)
            if attr is not None:
                info.external_attr = attr
            zf.writestr(info, content)
    return path


def test_zip_roundtrip(tmp_path):
    archive = _make_zip(
        tmp_path / "ok.zip",
        [
            ("skill/", b"", None),
            ("skill/SKILL.md", b"abc", None),
            ("skill/a/b.txt", b"12", 0o100644 << 16),
        ],
    )
    assert validate_zip_archive(archive) == 5
    out = extract_zip_archive(archive, tmp_path / "out")
    assert (out / "skill" / "SKILL.md").read_bytes() == b"abc"
    assert (out / "skill" / "a" / "b.txt").read_bytes() == b"12"


def test_zip_windows_style_directory_entry(tmp_path):
    # Archives built on Windows mark directories with the MS-DOS attribute and no slash.
    archive = _make_zip(tmp_path / "win.zip", [("d", b"", 0x10), ("d/f.txt", b"1", None)])
    assert validate_zip_archive(archive) == 1
    out = extract_zip_archive(archive, tmp_path / "out")
    assert (out / "d").is_dir()
    assert (out / "d" / "f.txt").read_bytes() == b"1"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([("../x.txt", b"x", None)], "unsafe path"),
        ([("/abs.txt", b"x", None)], "unsafe path"),
        ([("a\\b.txt", b"x", None)], "unsafe path"),
        ([("C:evil.txt", b"x", None)], "unsafe path"),
        ([("link", b"SKILL.md", (stat.S_IFLNK | 0o777) << 16)], "not a regular file or directory"),
        ([("fifo", b"", (stat.S_IFIFO | 0o644) << 16)], "not a regular file or directory"),
        ([("a.md", b"1", None), ("a.md", b"2", None)], "more than once"),
        ([(_NFD, b"1", None), (_NFC, b"2", None)], "normalize to the same path"),
        ([("A.md", b"1", None), ("a.md", b"2", None)], "differ only by letter case"),
        ([("x", b"1", None), ("x/y", b"2", None)], "as a directory, but it is a file"),
        ([("x/y", b"1", None), ("x", b"2", None)], "is also a directory"),
    ],
)
def test_validate_zip_archive_rejects_unsafe_entries(tmp_path, entries, message):
    archive = _make_zip(tmp_path / "bad.zip", entries)
    with pytest.raises(MlflowException, match=message):
        validate_zip_archive(archive)
    with pytest.raises(MlflowException, match=message):
        extract_zip_archive(archive, tmp_path / "out")


def test_validate_zip_archive_size_limit_and_bad_file(tmp_path):
    archive = _make_zip(tmp_path / "big.zip", [("a.bin", b"x" * 100, None)])
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_zip_archive(archive, max_bytes=99)
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"nope")
    with pytest.raises(MlflowException, match="not a readable ZIP archive"):
        validate_zip_archive(bogus)


def test_zip_subpath_limits_extraction_and_budget(tmp_path):
    archive = _make_zip(
        tmp_path / "pkg.zip",
        [("skills/a/SKILL.md", b"abc", None), ("other/big.bin", b"x" * 100, None)],
    )
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        validate_zip_archive(archive, max_bytes=10)
    assert validate_zip_archive(archive, max_bytes=10, subpath="skills/a") == 3
    out = extract_zip_archive(archive, tmp_path / "out", max_bytes=10, subpath="skills/a")
    assert (out / "skills" / "a" / "SKILL.md").read_bytes() == b"abc"
    assert not (out / "other").exists()
