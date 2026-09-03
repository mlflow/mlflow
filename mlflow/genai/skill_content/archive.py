from __future__ import annotations

import gzip
import os
import stat
import tarfile
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

from mlflow.environment_variables import MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.paths import (
    PathCollisionGuard,
    assert_regular_tree,
    canonical_relative_path,
    collect_tree,
    ensure_within,
    is_under_subpath,
    normalize_subpath,
)

# Bounds the number of archive entries so a tiny archive cannot force millions of file
# operations even when its declared content size is within the byte limit.
MAX_ARCHIVE_ENTRIES = 10_000
# Tar metadata (headers, padding, PAX and GNU long-name payloads) is read by the tar parser
# before any member is surfaced, so it is bounded separately from content by a fixed allowance.
_TAR_METADATA_ALLOWANCE = 1024 * MAX_ARCHIVE_ENTRIES
_COPY_CHUNK_SIZE = 1024 * 1024
_MSDOS_DIRECTORY_ATTRIBUTE = 0x10
_ZIP_TYPE_MASK = 0o170000


def get_max_decompressed_size(max_bytes: int | None = None) -> int:
    """
    The decompressed byte budget for skill content.

    ``max_bytes`` overrides the shared ``MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE`` setting
    for a single operation; every other caller shares the configured default.
    """
    limit = MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.get() if max_bytes is None else max_bytes
    if limit <= 0:
        raise invalid_content(f"Skill content size limit must be positive, got {limit}.")
    return limit


def _member_relative_path(name: str) -> str | None:
    """Canonicalize an exact archive entry name; ``None`` means the entry is the root itself."""
    value = name.removeprefix("./")
    if value in ("", ".", "./"):
        return None
    try:
        return canonical_relative_path(value)
    except Exception as e:
        raise invalid_content(f"Archive entry '{name}' has an unsafe path: {e}")


class _ByteBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def consume(self, amount: int, what: str) -> None:
        self.used += amount
        if self.used > self.limit:
            raise invalid_content(
                f"{what} exceeds the skill content size limit of {self.limit} bytes."
            )


class _BoundedStream:
    """Read-only wrapper that fails once more than ``limit`` decompressed bytes pass through."""

    def __init__(self, inner: IO[bytes], limit: int):
        self._inner = inner
        self._limit = limit
        self._read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._inner.read(size)
        self._read += len(chunk)
        if self._read > self._limit:
            raise invalid_content(
                f"Archive decompresses to more than {self._limit} bytes including metadata, "
                "which exceeds the skill content size limit."
            )
        return chunk

    def close(self) -> None:
        self._inner.close()


class _ArchiveLayout:
    """Tracks entries of one archive so files and directories cannot shadow each other."""

    def __init__(self):
        self._guard = PathCollisionGuard()
        self._files: set[str] = set()
        self._dirs: set[str] = set()

    def _register_dir(self, path: str, raw_name: str) -> None:
        if path in self._dirs:
            return
        if path in self._files:
            raise invalid_content(
                f"Archive entry '{raw_name}' uses '{path}' as a directory, but it is a file."
            )
        self._guard.register(path)
        self._dirs.add(path)

    def add(self, relative: str, raw_name: str, *, is_dir: bool) -> None:
        parts = relative.split("/")
        for depth in range(1, len(parts)):
            self._register_dir("/".join(parts[:depth]), raw_name)
        if is_dir:
            self._register_dir(relative, raw_name)
            return
        if relative in self._dirs:
            raise invalid_content(
                f"Archive entry '{raw_name}' is a file, but '{relative}' is also a directory."
            )
        self._guard.register(relative)
        self._files.add(relative)


def _copy_with_budget(src: IO[bytes], dst: IO[bytes], budget: _ByteBudget, what: str) -> None:
    while chunk := src.read(_COPY_CHUNK_SIZE):
        budget.consume(len(chunk), what)
        dst.write(chunk)


def _require_empty_dir(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise invalid_content(f"Extraction destination '{dest}' must be an empty directory.")


def _write_entry(
    dest: Path, relative: str, raw_name: str, source: IO[bytes] | None, budget: _ByteBudget
) -> None:
    target = dest.joinpath(*relative.split("/"))
    ensure_within(dest, target)
    try:
        if source is None:
            target.mkdir(parents=True, exist_ok=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, open(target, "wb") as out:
            _copy_with_budget(source, out, budget, "Archive content")
    except OSError as e:
        raise invalid_content(f"Cannot extract archive entry '{raw_name}': {e}")


# --- gzip tar ---------------------------------------------------------------------------------


def package_skill_tree(root: str | os.PathLike[str], output: str | os.PathLike[str]) -> Path:
    """
    Package the regular files under ``root`` as a reproducible gzip-compressed tar archive.

    The archive contains exactly the tree that ``compute_tree_digest`` hashes: the same file
    set in the same canonical order, with timestamps, ownership, and permissions normalized so
    identical content produces byte-identical archives on every platform.
    """
    files = collect_tree(root)
    output_path = Path(output)
    with (
        open(output_path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for entry in files:
            info = tarfile.TarInfo(name=entry.path)
            info.size = entry.size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(entry.local_path, "rb") as fh:
                tar.addfile(info, fh)
    return output_path


_TAR_FAILURES = (tarfile.TarError, EOFError, OSError, zlib.error, RecursionError, ValueError)


@contextmanager
def _open_tar(archive: Path, *, compressed: bool, limit: int) -> Iterator[tarfile.TarFile]:
    """
    Open ``archive`` as a sequential tar stream whose decompressed bytes are bounded.

    Streaming mode makes the parser read headers and content strictly in order, and the
    bounded stream underneath fails the moment the archive decompresses past the content
    limit plus a fixed metadata allowance, so oversized PAX or GNU long-name headers are cut
    off instead of being buffered in memory.
    """
    with open(archive, "rb") as raw:
        inner = gzip.GzipFile(fileobj=raw, mode="rb") if compressed else raw
        bounded = _BoundedStream(inner, limit + _TAR_METADATA_ALLOWANCE)
        try:
            with tarfile.open(fileobj=bounded, mode="r|") as tar:
                yield tar
        except _TAR_FAILURES as e:
            raise invalid_content(f"'{archive.name}' is not a readable tar archive: {e}")


def _iter_tar_members(tar: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    count = 0
    try:
        for member in tar:
            count += 1
            if count > MAX_ARCHIVE_ENTRIES:
                raise invalid_content(f"Archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
            yield member
    except _TAR_FAILURES as e:
        raise invalid_content(f"Archive is malformed: {e}")


def _tar_entry_kind(member: tarfile.TarInfo) -> bool:
    """Return ``is_dir`` for an accepted entry, or raise for any other entry type."""
    if member.isdir():
        return True
    if member.isreg():
        return False
    raise invalid_content(f"Archive entry '{member.name}' is not a regular file or directory.")


def validate_skill_archive(
    archive: str | os.PathLike[str],
    *,
    max_bytes: int | None = None,
    compressed: bool = True,
    subpath: str | None = None,
) -> int:
    """
    Validate a skill archive without extracting it.

    Every entry is checked: absolute paths, ``..`` segments, backslashes, drive or stream
    colons, reserved names, entries that are not regular files or directories, and names that
    collide after Unicode normalization or case folding are rejected. When ``subpath`` is
    given, only entries at or beneath it count toward the decompressed size limit, because
    only those are extracted.

    Returns:
        The total declared size of the regular files that would be extracted.
    """
    limit = get_max_decompressed_size(max_bytes)
    prefix = normalize_subpath(subpath)
    budget = _ByteBudget(limit)
    layout = _ArchiveLayout()
    with _open_tar(Path(archive), compressed=compressed, limit=limit) as tar:
        for member in _iter_tar_members(tar):
            relative = _member_relative_path(member.name)
            is_dir = _tar_entry_kind(member)
            if relative is None:
                if not is_dir:
                    raise invalid_content("Archive root entry must be a directory.")
                continue
            layout.add(relative, member.name, is_dir=is_dir)
            if not is_dir and is_under_subpath(relative, prefix):
                budget.consume(member.size, "Archive content")
    return budget.used


def extract_skill_archive(
    archive: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    max_bytes: int | None = None,
    compressed: bool = True,
    subpath: str | None = None,
) -> Path:
    """
    Validate ``archive`` and extract it into the empty directory ``dest``.

    Only regular files and directories are written, the decompressed size limit is enforced on
    the bytes actually read, and the result is verified with ``assert_regular_tree``. When
    ``subpath`` is given, entries outside it are validated but not written, so the extracted
    tree contains just ``dest/<subpath>``.
    """
    archive_path = Path(archive)
    dest_path = Path(dest)
    limit = get_max_decompressed_size(max_bytes)
    prefix = normalize_subpath(subpath)
    validate_skill_archive(archive_path, max_bytes=limit, compressed=compressed, subpath=prefix)
    _require_empty_dir(dest_path)
    budget = _ByteBudget(limit)
    with _open_tar(archive_path, compressed=compressed, limit=limit) as tar:
        for member in _iter_tar_members(tar):
            relative = _member_relative_path(member.name)
            if relative is None or not is_under_subpath(relative, prefix):
                continue
            if member.isdir():
                _write_entry(dest_path, relative, member.name, None, budget)
                continue
            source = tar.extractfile(member)
            if source is None:
                raise invalid_content(f"Archive entry '{member.name}' has no readable content.")
            _write_entry(dest_path, relative, member.name, source, budget)
    assert_regular_tree(dest_path)
    return dest_path


# --- zip --------------------------------------------------------------------------------------


def _zip_is_dir(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & _ZIP_TYPE_MASK
    return (
        info.is_dir()
        or mode == stat.S_IFDIR
        or bool(info.external_attr & _MSDOS_DIRECTORY_ATTRIBUTE)
    )


def _zip_entry_kind(info: zipfile.ZipInfo) -> bool:
    if _zip_is_dir(info):
        return True
    mode = (info.external_attr >> 16) & _ZIP_TYPE_MASK
    if mode in (0, stat.S_IFREG):
        return False
    raise invalid_content(f"Archive entry '{info.filename}' is not a regular file or directory.")


@contextmanager
def _open_zip(archive: Path) -> Iterator[zipfile.ZipFile]:
    try:
        with zipfile.ZipFile(archive) as zf:
            yield zf
    except (zipfile.BadZipFile, EOFError, OSError, zlib.error) as e:
        raise invalid_content(f"'{archive.name}' is not a readable ZIP archive: {e}")


def validate_zip_archive(
    archive: str | os.PathLike[str], *, max_bytes: int | None = None, subpath: str | None = None
) -> int:
    """Validate a ZIP archive with the same path, entry type, collision, and size rules as tar."""
    prefix = normalize_subpath(subpath)
    budget = _ByteBudget(get_max_decompressed_size(max_bytes))
    layout = _ArchiveLayout()
    with _open_zip(Path(archive)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise invalid_content(f"Archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
        for info in infos:
            relative = _member_relative_path(info.filename)
            is_dir = _zip_entry_kind(info)
            if relative is None:
                if not is_dir:
                    raise invalid_content("Archive root entry must be a directory.")
                continue
            layout.add(relative, info.filename, is_dir=is_dir)
            if not is_dir and is_under_subpath(relative, prefix):
                budget.consume(info.file_size, "Archive content")
    return budget.used


def extract_zip_archive(
    archive: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    max_bytes: int | None = None,
    subpath: str | None = None,
) -> Path:
    """Validate ``archive`` and extract it, or only ``subpath``, into empty directory ``dest``."""
    dest_path = Path(dest)
    limit = get_max_decompressed_size(max_bytes)
    prefix = normalize_subpath(subpath)
    validate_zip_archive(archive, max_bytes=limit, subpath=prefix)
    _require_empty_dir(dest_path)
    budget = _ByteBudget(limit)
    with _open_zip(Path(archive)) as zf:
        for info in zf.infolist():
            relative = _member_relative_path(info.filename)
            if relative is None or not is_under_subpath(relative, prefix):
                continue
            if _zip_is_dir(info):
                _write_entry(dest_path, relative, info.filename, None, budget)
                continue
            try:
                source = zf.open(info)
            except (zipfile.BadZipFile, zlib.error, OSError) as e:
                raise invalid_content(f"Cannot extract archive entry '{info.filename}': {e}")
            _write_entry(dest_path, relative, info.filename, source, budget)
    assert_regular_tree(dest_path)
    return dest_path
