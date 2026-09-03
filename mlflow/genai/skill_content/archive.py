from __future__ import annotations

import gzip
import os
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import IO

from mlflow.environment_variables import MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.paths import assert_regular_tree, collect_tree, normalize_subpath

# Bounds the number of archive entries so a tiny archive cannot force millions of file
# operations even when its declared content size is within the byte limit.
MAX_ARCHIVE_ENTRIES = 10_000
_COPY_CHUNK_SIZE = 1024 * 1024


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
    """Canonicalize an archive entry name; ``None`` means the entry is the root itself."""
    value = name
    value = value.removeprefix("./")
    if value in ("", "."):
        return None
    try:
        return normalize_subpath(value)
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


def _copy_with_budget(src: IO[bytes], dst: IO[bytes], budget: _ByteBudget, what: str) -> None:
    while chunk := src.read(_COPY_CHUNK_SIZE):
        budget.consume(len(chunk), what)
        dst.write(chunk)


def _require_empty_dir(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise invalid_content(f"Extraction destination '{dest}' must be an empty directory.")


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


def _open_tar(archive: Path, *, compressed: bool) -> tarfile.TarFile:
    try:
        return tarfile.open(archive, "r:gz" if compressed else "r:")
    except (tarfile.ReadError, OSError, EOFError) as e:
        raise invalid_content(f"'{archive.name}' is not a readable tar archive: {e}")


def _iter_tar_members(tar: tarfile.TarFile):
    count = 0
    try:
        for member in tar:
            count += 1
            if count > MAX_ARCHIVE_ENTRIES:
                raise invalid_content(f"Archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
            yield member
    except (tarfile.TarError, EOFError) as e:
        raise invalid_content(f"Archive is malformed: {e}")


def validate_skill_archive(
    archive: str | os.PathLike[str], *, max_bytes: int | None = None, compressed: bool = True
) -> int:
    """
    Validate a skill archive without extracting it.

    Rejects absolute paths, ``..`` segments, backslashes, and any entry that is not a regular
    file or directory (symbolic links, hard links, devices, FIFOs). The sum of the declared
    entry sizes must not exceed the decompressed size limit.

    Returns:
        The total declared size of the regular files in the archive.
    """
    limit = get_max_decompressed_size(max_bytes)
    budget = _ByteBudget(limit)
    with _open_tar(Path(archive), compressed=compressed) as tar:
        for member in _iter_tar_members(tar):
            relative = _member_relative_path(member.name)
            if member.isdir():
                continue
            if relative is None:
                raise invalid_content("Archive root entry must be a directory.")
            if not member.isreg():
                raise invalid_content(
                    f"Archive entry '{member.name}' is not a regular file or directory."
                )
            budget.consume(member.size, "Archive content")
    return budget.used


def extract_skill_archive(
    archive: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    max_bytes: int | None = None,
    compressed: bool = True,
) -> Path:
    """
    Validate ``archive`` and extract it into the empty directory ``dest``.

    Extraction writes regular files and directories only, enforces the decompressed size
    limit on the bytes actually read, and verifies the result with ``assert_regular_tree``.
    """
    archive_path = Path(archive)
    dest_path = Path(dest)
    validate_skill_archive(archive_path, max_bytes=max_bytes, compressed=compressed)
    _require_empty_dir(dest_path)
    budget = _ByteBudget(get_max_decompressed_size(max_bytes))
    with _open_tar(archive_path, compressed=compressed) as tar:
        for member in _iter_tar_members(tar):
            relative = _member_relative_path(member.name)
            if relative is None:
                continue
            target = dest_path.joinpath(*relative.split("/"))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise invalid_content(f"Archive entry '{member.name}' has no readable content.")
            try:
                with source, open(target, "wb") as out:
                    _copy_with_budget(source, out, budget, "Archive content")
            except OSError as e:
                raise invalid_content(f"Cannot extract archive entry '{member.name}': {e}")
    assert_regular_tree(dest_path)
    return dest_path


# --- zip --------------------------------------------------------------------------------------

_ZIP_TYPE_MASK = 0o170000


def _zip_entry_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & _ZIP_TYPE_MASK


def validate_zip_archive(archive: str | os.PathLike[str], *, max_bytes: int | None = None) -> int:
    """Validate a ZIP archive with the same path, entry type, and size rules as tar archives."""
    budget = _ByteBudget(get_max_decompressed_size(max_bytes))
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                raise invalid_content(f"Archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
            for info in infos:
                relative = _member_relative_path(info.filename)
                if info.is_dir():
                    continue
                if relative is None:
                    raise invalid_content("Archive root entry must be a directory.")
                mode = _zip_entry_mode(info)
                if mode not in (0, stat.S_IFREG):
                    raise invalid_content(
                        f"Archive entry '{info.filename}' is not a regular file or directory."
                    )
                budget.consume(info.file_size, "Archive content")
    except zipfile.BadZipFile as e:
        raise invalid_content(f"'{Path(archive).name}' is not a readable ZIP archive: {e}")
    return budget.used


def extract_zip_archive(
    archive: str | os.PathLike[str], dest: str | os.PathLike[str], *, max_bytes: int | None = None
) -> Path:
    """Validate ``archive`` and extract it into the empty directory ``dest``."""
    dest_path = Path(dest)
    validate_zip_archive(archive, max_bytes=max_bytes)
    _require_empty_dir(dest_path)
    budget = _ByteBudget(get_max_decompressed_size(max_bytes))
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            relative = _member_relative_path(info.filename)
            if relative is None:
                continue
            target = dest_path.joinpath(*relative.split("/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as source, open(target, "wb") as out:
                    _copy_with_budget(source, out, budget, "Archive content")
            except (OSError, zipfile.BadZipFile) as e:
                raise invalid_content(f"Cannot extract archive entry '{info.filename}': {e}")
    assert_regular_tree(dest_path)
    return dest_path
