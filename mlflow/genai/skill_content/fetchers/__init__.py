from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from mlflow.entities.skill_source import SkillSourceType
from mlflow.genai.skill_content.archive import get_max_decompressed_size
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.fetchers.artifacts import fetch_mlflow_artifacts
from mlflow.genai.skill_content.fetchers.git import fetch_git
from mlflow.genai.skill_content.fetchers.oci import fetch_oci
from mlflow.genai.skill_content.fetchers.zip import fetch_zip
from mlflow.genai.skill_content.paths import assert_regular_tree, resolve_contained, tree_size
from mlflow.genai.skill_content.sources import ResolvedSource, SourceInput, resolve_source_type


@dataclass
class FetchedContent:
    """
    Skill content materialized on the local filesystem.

    ``root`` is the directory the caller should inspect or hash: the fetched tree after the
    subpath has been applied. Remote fetches live in a temporary directory that ``cleanup``
    removes; local sources are used in place and ``cleanup`` is a no-op for them.
    """

    root: Path
    resolved: ResolvedSource
    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> FetchedContent:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def _fetch_remote(resolved: ResolvedSource, dest: Path, limit: int) -> Path:
    subpath = resolved.subpath
    if resolved.source_type == SkillSourceType.GIT:
        return fetch_git(resolved.source, resolved.ref, dest, max_bytes=limit, subpath=subpath)
    if resolved.source_type == SkillSourceType.OCI:
        return fetch_oci(resolved.source, dest, max_bytes=limit, subpath=subpath)
    if resolved.source_type == SkillSourceType.ZIP:
        return fetch_zip(resolved.source, dest, max_bytes=limit, subpath=subpath)
    if resolved.source_type == SkillSourceType.MLFLOW:
        return fetch_mlflow_artifacts(resolved.source, dest, max_bytes=limit, subpath=subpath)
    raise invalid_content(f"Source type '{resolved.source_type}' cannot be fetched.")


def fetch_source(
    source: SourceInput,
    *,
    ref: str | None = None,
    subpath: str | None = None,
    max_bytes: int | None = None,
) -> FetchedContent:
    """
    Fetch skill content from a local path or a Git, OCI, ZIP, or MLflow artifact source.

    Fetching uses the caller's own credentials (Git credential helpers, Docker config and
    credential helpers, MLflow tracking credentials); ZIP sources must be publicly reachable.
    The decompressed size limit bounds the content at or beneath ``subpath``: archives and
    images extract only that part, and Git checkouts are measured there. Downloads themselves
    are bounded by the same limit on the wire so an oversized archive is cut off early. After
    the fetch, the subpath is resolved with containment checks and the resulting tree is
    required to contain only regular files and directories.

    Args:
        source: Typed source, remote URL or image reference, MLflow artifact URI, or local path.
        ref: Git branch, tag, or commit for a plain-string Git URL. Typed sources carry their own.
        subpath: Directory within the fetched content to use as the root.
        max_bytes: Per-call override of the shared decompressed size limit.

    Returns:
        A ``FetchedContent`` whose ``root`` is ready for inspection, hashing, or packaging.
    """
    resolved = resolve_source_type(source, ref=ref, subpath=subpath)
    limit = get_max_decompressed_size(max_bytes)
    if resolved.is_local:
        base = Path(resolved.source)
        if not base.is_dir():
            raise invalid_content(f"Local skill source '{base}' is not a directory.")
        tmpdir = None
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="mlflow-skill-content-")
        dest = Path(tmpdir.name) / "content"
        try:
            base = _fetch_remote(resolved, dest, limit)
        except Exception:
            tmpdir.cleanup()
            raise
    try:
        root = resolve_contained(base, resolved.subpath)
        assert_regular_tree(root)
        if (size := tree_size(root)) > limit:
            raise invalid_content(
                f"Skill content is {size} bytes, which exceeds the size limit of {limit} bytes."
            )
    except Exception:
        if tmpdir is not None:
            tmpdir.cleanup()
        raise
    return FetchedContent(root=root, resolved=resolved, _tmpdir=tmpdir)
