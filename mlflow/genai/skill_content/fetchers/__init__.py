from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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
    subpath has been applied. Remote fetches use a temporary directory that ``cleanup``
    removes; local sources are used in place, and content fetched into a caller-supplied
    destination stays there, so ``cleanup`` removes only the scratch space for those.
    """

    root: Path
    resolved: ResolvedSource
    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


def _fetch_remote(resolved: ResolvedSource, dest: Path, scratch: Path, limit: int) -> Path:
    subpath = resolved.subpath
    if resolved.source_type == SkillSourceType.GIT:
        return fetch_git(
            resolved.source, resolved.ref, dest, scratch=scratch, max_bytes=limit, subpath=subpath
        )
    if resolved.source_type == SkillSourceType.OCI:
        return fetch_oci(resolved.source, dest, scratch=scratch, max_bytes=limit, subpath=subpath)
    if resolved.source_type == SkillSourceType.ZIP:
        return fetch_zip(resolved.source, dest, scratch=scratch, max_bytes=limit, subpath=subpath)
    if resolved.source_type == SkillSourceType.MLFLOW:
        return fetch_mlflow_artifacts(resolved.source, dest, max_bytes=limit, subpath=subpath)
    raise invalid_content(f"Source type '{resolved.source_type}' cannot be fetched.")


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise invalid_content(f"Destination '{destination}' is not a directory.")
        if any(destination.iterdir()):
            raise invalid_content(f"Destination '{destination}' must be empty.")
    else:
        destination.mkdir(parents=True)


def _clear_destination(destination: Path) -> None:
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _validated_root(base: Path, subpath: str | None, limit: int) -> Path:
    root = resolve_contained(base, subpath)
    assert_regular_tree(root)
    if (size := tree_size(root)) > limit:
        raise invalid_content(
            f"Skill content is {size} bytes, which exceeds the size limit of {limit} bytes."
        )
    return root


@contextmanager
def fetch_source(
    source: SourceInput,
    *,
    ref: str | None = None,
    subpath: str | None = None,
    max_bytes: int | None = None,
    destination: Path | None = None,
) -> Iterator[FetchedContent]:
    """
    Fetch skill content from a local path or a Git, OCI, ZIP, or MLflow artifact source.

    Fetching uses the caller's own credentials (Git credential helpers, Docker config and
    credential helpers, MLflow tracking credentials); ZIP sources must be publicly reachable.
    The decompressed size limit bounds the content at or beneath ``subpath``: archives and
    images extract only that part, and Git trees are measured there. Downloads themselves are
    bounded by the same limit on the wire so an oversized archive is cut off early. After the
    fetch, the subpath is resolved with containment checks and the resulting tree is required
    to contain only regular files and directories.

    Used as a context manager: remote content lives in a temporary directory that is removed
    when the block exits, unless ``destination`` is given, in which case the content is written
    there and only the scratch space (Git objects, downloaded archives) is temporary. Local
    sources are used in place and never copied.

    Args:
        source: Typed source, remote URL or image reference, MLflow artifact URI, or local path.
        ref: Git branch, tag, or commit for a plain-string Git URL. Typed sources carry their own.
        subpath: Directory within the fetched content to use as the root.
        max_bytes: Per-call override of the decompressed size limit. Defaults to
            ``MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE``, which is 25 MiB unless configured.
        destination: Directory to fetch remote content into instead of a temporary one. It must
            not exist yet or must be empty, and is left in place on success. Not applicable to
            local sources, which are used where they are.

    Yields:
        A ``FetchedContent`` whose ``root`` is ready for inspection, hashing, or packaging.
    """
    resolved = resolve_source_type(source, ref=ref, subpath=subpath)
    limit = get_max_decompressed_size(max_bytes)
    if resolved.is_local:
        if destination is not None:
            raise invalid_content(
                "'destination' applies to remote sources only; a local source is used in place."
            )
        base = Path(resolved.source)
        if not base.is_dir():
            raise invalid_content(f"Local skill source '{base}' is not a directory.")
        yield FetchedContent(root=_validated_root(base, resolved.subpath, limit), resolved=resolved)
        return

    if destination is not None:
        _prepare_destination(destination)
    tmpdir = tempfile.TemporaryDirectory(prefix="mlflow-skill-content-")
    scratch = Path(tmpdir.name)
    dest = scratch / "content" if destination is None else destination
    fetched = None
    try:
        base = _fetch_remote(resolved, dest, scratch, limit)
        root = _validated_root(base, resolved.subpath, limit)
        fetched = FetchedContent(root=root, resolved=resolved, _tmpdir=tmpdir)
        yield fetched
    except BaseException:
        if fetched is None and destination is not None:
            _clear_destination(destination)
        raise
    finally:
        if fetched is not None:
            fetched.cleanup()
        else:
            tmpdir.cleanup()
