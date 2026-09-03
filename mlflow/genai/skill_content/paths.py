from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


def normalize_subpath(subpath: str | None) -> str | None:
    """
    Normalize a path within a fetched source to a POSIX relative path.

    Returns ``None`` for an empty subpath (the whole tree). Rejects backslashes, absolute paths,
    and ``.`` or ``..`` segments so a subpath can never escape or alias the content root.
    """
    if subpath is None:
        return None
    if not isinstance(subpath, str):
        raise invalid_content(f"Subpath must be a string, got {type(subpath).__name__}.")
    value = subpath.strip()
    if value == "":
        return None
    if "\\" in value:
        raise invalid_content(f"Subpath '{subpath}' must use forward slashes as separators.")
    if value.startswith("/"):
        raise invalid_content(f"Subpath '{subpath}' must be relative to the content root.")
    parts = [part for part in value.split("/") if part != ""]
    if any(part in (".", "..") for part in parts):
        raise invalid_content(f"Subpath '{subpath}' must not contain '.' or '..' segments.")
    return "/".join(parts) if parts else None


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
    return current


@dataclass(frozen=True)
class TreeFile:
    """A regular file within a skill tree, keyed by its canonical POSIX path."""

    path: str
    local_path: Path
    size: int


def _canonical_relative_path(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root).as_posix()
    return unicodedata.normalize("NFC", relative)


def collect_tree(root: str | os.PathLike[str]) -> list[TreeFile]:
    """
    List the regular files under ``root`` in canonical digest order.

    Paths are POSIX, relative to ``root``, Unicode NFC-normalized, and sorted by their UTF-8
    byte value. Symbolic links and other non-regular entries are excluded. Two files whose
    names normalize to the same path make the tree ambiguous and are rejected.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise invalid_content(f"Content root '{root_path}' is not a directory.")
    seen: dict[str, str] = {}
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
            canonical = _canonical_relative_path(root_path, file_path)
            raw = file_path.relative_to(root_path).as_posix()
            if (previous := seen.get(canonical)) is not None:
                raise invalid_content(
                    f"Files '{previous}' and '{raw}' normalize to the same path '{canonical}'; "
                    "the skill tree is ambiguous."
                )
            seen[canonical] = raw
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
