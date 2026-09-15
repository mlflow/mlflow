from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import IO

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import (
    display_path,
    error_code_for_http_status,
    invalid_content,
    source_unavailable,
)
from mlflow.genai.skill_content.paths import (
    TreeLayout,
    canonical_relative_path,
    ensure_within,
    normalize_subpath,
)
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
)

_GIT_TIMEOUT_SECONDS = 600
_COPY_CHUNK_SIZE = 1024 * 1024
_HTTP_STATUS_PATTERN = re.compile(r"(?:returned error|http status|status code|error):?\s*(\d{3})\b")


def _git_environment(no_hooks_dir: Path) -> dict[str, str]:
    """
    Environment for the fetch command.

    ``GIT_TERMINAL_PROMPT=0`` keeps git from blocking on an interactive credential prompt;
    credentials still come from the caller's helpers, SSH agent, or netrc. The ``GIT_CONFIG_*``
    entries override every config scope, including the caller's global one: ``core.hooksPath``
    points at an empty directory so nothing shipped in a repository can ever run as a hook.
    No worktree is checked out, so this is defense in depth rather than the only barrier.
    """
    return {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(no_hooks_dir),
    }


_AUTH_MARKERS = (
    "authentication failed",
    "could not read username",
    "could not read password",
    "permission denied",
    "publickey",
    "terminal prompts disabled",
)
_AVAILABILITY_MARKERS = (
    "could not resolve host",
    "connection refused",
    "connection timed out",
    "timed out",
    "unable to access",
    "early eof",
)


def _error_code_for_git(detail: str) -> int:
    lowered = detail.lower()
    # HTTP transports often report only the status, e.g. "returned error: 403".
    if status := _HTTP_STATUS_PATTERN.search(lowered):
        return error_code_for_http_status(int(status.group(1)))
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return UNAUTHENTICATED
    if any(marker in lowered for marker in _AVAILABILITY_MARKERS):
        return TEMPORARILY_UNAVAILABLE
    return RESOURCE_DOES_NOT_EXIST


def _copy_stream(source: IO[bytes], target: Path) -> None:
    with open(target, "wb") as out:
        while chunk := source.read(_COPY_CHUNK_SIZE):
            out.write(chunk)


def _materialize_tree(tree, dest: Path, *, max_bytes: int) -> None:
    """
    Write the committed blobs of ``tree`` into ``dest`` without a worktree checkout.

    Reading objects directly means none of git's working-tree conversions apply: no smudge or
    process filter selected by the repository's ``.gitattributes``, no ``core.autocrlf`` or
    ``core.eol`` rewriting, and no symlink creation. Every path passes the same segment and
    layout rules as archive entries, symbolic links and submodules are rejected, and the size
    limit applies to the bytes written.
    """
    layout = TreeLayout()
    written = 0
    for item in tree.traverse():
        relative = canonical_relative_path(item.path)
        if relative is None:
            continue
        target = dest.joinpath(*relative.split("/"))
        ensure_within(dest, target)
        if item.type == "tree":
            layout.add(relative, item.path, is_dir=True)
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.type == "commit":
            raise invalid_content(
                f"Git source contains a submodule at '{display_path(item.path)}'; submodules "
                "are not supported."
            )
        if item.type != "blob":
            continue
        if stat.S_ISLNK(item.mode):
            raise invalid_content(
                f"Skill content must not contain symbolic links: '{display_path(item.path)}'."
            )
        layout.add(relative, item.path, is_dir=False)
        written += item.size
        if written > max_bytes:
            raise invalid_content(
                f"Git content is at least {written} bytes, which exceeds the skill content "
                f"size limit of {max_bytes} bytes."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_stream(item.data_stream, target)
        if item.mode & stat.S_IXUSR:
            target.chmod(0o755)


def fetch_git(
    url: str,
    ref: str | None,
    dest: Path,
    *,
    scratch: Path,
    max_bytes: int,
    subpath: str | None = None,
) -> Path:
    """
    Materialize the tree at ``ref`` of the repository ``url`` under ``dest``.

    A shallow fetch of the single ref (or the remote ``HEAD`` when ``ref`` is omitted) brings
    the objects into a repository under ``scratch``; the committed blobs at ``subpath``
    are then written straight from the object store, so the caller's checkout settings and the
    repository's attributes never touch the bytes. Submodules are not supported; a skill is a
    plain content tree. The size limit applies to the tree at ``subpath``.
    """
    # GitPython needs the git executable at import time, so import only when fetching.
    import git

    prefix = normalize_subpath(subpath)
    dest.mkdir(parents=True, exist_ok=True)
    objects = scratch / "git-objects"
    no_hooks_dir = scratch / "no-hooks"
    no_hooks_dir.mkdir(parents=True, exist_ok=True)
    target = f"{url} at ref '{ref}'" if ref else url
    repo = git.Repo.init(objects)
    try:
        with repo.git.custom_environment(**_git_environment(no_hooks_dir)):
            origin = repo.create_remote("origin", url)
            # A partial fetch brings only commits and trees; blobs are retrieved on demand as
            # they are read, so a small skill in a large repository transfers only its own
            # files. Servers without partial-clone support ignore the filter and send all.
            origin.fetch(
                refspec=ref or "HEAD",
                depth=1,
                filter="blob:none",
                kill_after_timeout=_GIT_TIMEOUT_SECONDS,
            )
        try:
            tree = repo.commit("FETCH_HEAD").tree
        except (git.exc.BadName, ValueError) as e:
            raise source_unavailable(target, f"fetched ref has no commit: {e}")
        if prefix is not None:
            try:
                tree = tree / prefix
            except KeyError:
                raise MlflowException(
                    f"Subpath '{prefix}' does not exist in the fetched content.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            if tree.type != "tree":
                raise invalid_content(f"Subpath '{prefix}' must point to a directory.")
            dest.joinpath(*prefix.split("/")).mkdir(parents=True, exist_ok=True)
        _materialize_tree(tree, dest, max_bytes=max_bytes)
    except git.exc.GitCommandError as e:
        detail = (e.stderr or str(e)).strip()
        raise source_unavailable(target, detail, error_code=_error_code_for_git(detail))
    finally:
        repo.close()
    return dest
