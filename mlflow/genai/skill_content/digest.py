from __future__ import annotations

import hashlib
import os
import struct

from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.paths import collect_tree

_READ_CHUNK_SIZE = 1024 * 1024


def _u64be(value: int) -> bytes:
    return struct.pack(">Q", value)


def compute_tree_digest(root: str | os.PathLike[str]) -> str:
    """
    Compute the canonical SHA-256 content digest of a skill tree.

    Regular files are visited in canonical order (see ``collect_tree``) and each contributes
    four length-framed parts: the UTF-8 length of its POSIX relative path as an unsigned 8-byte
    big-endian integer, the path bytes, the content length in the same encoding, and the
    content bytes. Only paths and contents contribute, so the same tree hashes identically
    regardless of where it was fetched from or which operating system hashed it.

    Returns:
        Lowercase 64-character hexadecimal digest.
    """
    hasher = hashlib.sha256()
    for entry in collect_tree(root):
        path_bytes = entry.path.encode("utf-8")
        hasher.update(_u64be(len(path_bytes)))
        hasher.update(path_bytes)
        hasher.update(_u64be(entry.size))
        read = 0
        with open(entry.local_path, "rb") as fh:
            while chunk := fh.read(_READ_CHUNK_SIZE):
                hasher.update(chunk)
                read += len(chunk)
        if read != entry.size:
            raise invalid_content(
                f"File '{entry.path}' changed while it was being hashed "
                f"(expected {entry.size} bytes, read {read})."
            )
    return hasher.hexdigest()
