from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

# Names Windows refuses to create as regular files, with or without an extension. A tree
# containing them cannot be materialized on every supported OS, so it is rejected everywhere.
_WINDOWS_RESERVED_NAMES = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def _validate_segment(segment: str, original: str) -> None:
    if segment in (".", ".."):
        raise invalid_content(f"Path '{original}' must not contain '.' or '..' segments.")
    if ":" in segment:
        # Drive letters (``C:``) and NTFS alternate data streams (``file:stream``) both make a
        # segment escape or alias the tree on Windows.
        raise invalid_content(f"Path '{original}' must not contain ':' in a segment.")
    if any(ord(ch) < 32 or ch == "\x7f" for ch in segment):
        raise invalid_content(f"Path '{original}' must not contain control characters.")
    if segment.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise invalid_content(f"Path '{original}' uses a reserved Windows device name.")


def canonical_relative_path(value: str) -> str | None:
    """
    Validate an exact relative path and return it in canonical POSIX form.

    Unlike ``normalize_subpath`` this never trims whitespace, so an archive entry named
    ``"a.md "`` keeps its exact name and the extracted tree stays byte-identical to the tree
    that was hashed. Returns ``None`` when the path names the root itself.
    """
    if "\\" in value:
        raise invalid_content(f"Path '{value}' must use forward slashes as separators.")
    if value.startswith("/"):
        raise invalid_content(f"Path '{value}' must be relative to the content root.")
    parts = [part for part in value.split("/") if part != ""]
    for part in parts:
        _validate_segment(part, value)
    return "/".join(parts) if parts else None


def normalize_subpath(subpath: str | None) -> str | None:
    """
    Normalize a caller-supplied path within a fetched source to a POSIX relative path.

    Returns ``None`` for an empty subpath (the whole tree). Surrounding whitespace is trimmed
    because this is user input; backslashes, absolute paths, ``.`` or ``..`` segments, drive
    or stream colons, and Windows reserved names are rejected so a subpath can never escape
    or alias the content root on any supported OS.
    """
    if subpath is None:
        return None
    if not isinstance(subpath, str):
        raise invalid_content(f"Subpath must be a string, got {type(subpath).__name__}.")
    value = subpath.strip()
    if value == "":
        return None
    try:
        return canonical_relative_path(value)
    except MlflowException as e:
        raise invalid_content(f"Subpath '{subpath}' is invalid: {e.message}")


def is_under_subpath(relative: str, subpath: str | None) -> bool:
    """Whether canonical path ``relative`` is ``subpath`` itself or lies beneath it."""
    if subpath is None:
        return True
    return relative == subpath or relative.startswith(subpath + "/")


def ensure_within(root: Path, target: Path) -> Path:
    """Defence in depth: prove that ``target`` resolves inside ``root`` before touching it."""
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_root and not resolved_target.is_relative_to(resolved_root):
        raise invalid_content(f"Path '{target}' resolves outside of '{root}'.")
    return resolved_target


def resolve_contained(root: str | os.PathLike[str], subpath: str | None) -> Path:
    """
    Resolve ``subpath`` under ``root`` and prove containment.

    Every segment is checked as it is traversed so a symbolic link inside the tree cannot
    redirect the lookup outside of ``root``. The resolved path must be an existing directory.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise invalid_content(f"Content root '{root_path}' is not a directory.")
    normalized = normalize_subpath(subpath)
    if normalized is None:
        return root_path
    current = root_path
    for segment in normalized.split("/"):
        current = current / segment
        if current.is_symlink():
            raise invalid_content(f"Subpath '{normalized}' traverses a symbolic link.")
        if not current.exists():
            raise MlflowException(
                f"Subpath '{normalized}' does not exist in the fetched content.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
    if not current.is_dir():
        raise invalid_content(f"Subpath '{normalized}' must point to a directory.")
    ensure_within(root_path, current)
    return current


@dataclass(frozen=True)
class TreeFile:
    """A regular file within a skill tree, keyed by its canonical POSIX path."""

    path: str
    local_path: Path
    size: int


class PathCollisionGuard:
    """
    Rejects paths that would alias each other on some supported filesystem.

    Two names collide when they are byte-identical, differ only by Unicode normalization
    form, or differ only by letter case. Any of these would make the same tree materialize
    differently on macOS, Linux, and Windows, so the tree is rejected as ambiguous instead.
    """

    def __init__(self):
        self._by_nfc: dict[str, str] = {}
        self._by_casefold: dict[str, str] = {}

    def register(self, raw_path: str) -> str:
        nfc = unicodedata.normalize("NFC", raw_path)
        if (previous := self._by_nfc.get(nfc)) is not None:
            if previous == raw_path:
                raise invalid_content(f"Path '{raw_path}' appears more than once.")
            raise invalid_content(
                f"Paths '{previous}' and '{raw_path}' normalize to the same path '{nfc}'; "
                "the skill tree is ambiguous."
            )
        folded = nfc.casefold()
        if (previous := self._by_casefold.get(folded)) is not None:
            raise invalid_content(
                f"Paths '{previous}' and '{raw_path}' differ only by letter case; "
                "the skill tree is ambiguous on case-insensitive filesystems."
            )
        self._by_nfc[nfc] = raw_path
        self._by_casefold[folded] = raw_path
        return nfc


def collect_tree(root: str | os.PathLike[str]) -> list[TreeFile]:
    """
    List the regular files under ``root`` in canonical digest order.

    Paths are POSIX, relative to ``root``, Unicode NFC-normalized, and sorted by their UTF-8
    byte value. Symbolic links and other non-regular entries are excluded. Files whose names
    collide after normalization or case folding make the tree ambiguous and are rejected.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise invalid_content(f"Content root '{root_path}' is not a directory.")
    guard = PathCollisionGuard()
    files: list[TreeFile] = []
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        current = Path(dirpath)
        # Do not descend into symlinked directories; os.walk lists them but must not follow.
        dirnames[:] = sorted(d for d in dirnames if not (current / d).is_symlink())
        for filename in filenames:
            file_path = current / filename
            info = os.lstat(file_path)
            if not stat.S_ISREG(info.st_mode):
                continue
            canonical = guard.register(file_path.relative_to(root_path).as_posix())
            files.append(TreeFile(path=canonical, local_path=file_path, size=info.st_size))
    files.sort(key=lambda f: f.path.encode("utf-8"))
    return files


def assert_regular_tree(root: str | os.PathLike[str]) -> None:
    """
    Require that ``root`` contains only directories and regular files.

    Symbolic links, hard links, and special files (FIFOs, sockets, devices) are rejected. This
    is the publication-time check applied to every fetched or extracted skill tree.
    """
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise invalid_content(f"Content root '{root_path}' must be a directory, not a link.")
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        current = Path(dirpath)
        for name in dirnames + filenames:
            entry = current / name
            info = os.lstat(entry)
            relative = entry.relative_to(root_path).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise invalid_content(
                    f"Skill content must not contain symbolic links: '{relative}'."
                )
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise invalid_content(
                    f"Skill content must contain only regular files and directories: '{relative}'."
                )
            if info.st_nlink > 1:
                raise invalid_content(f"Skill content must not contain hard links: '{relative}'.")


def tree_size(root: str | os.PathLike[str]) -> int:
    """Total size in bytes of the regular files under ``root``."""
    return sum(f.size for f in collect_tree(root))
