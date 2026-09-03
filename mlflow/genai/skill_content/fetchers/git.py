from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import resolve_contained, tree_size
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
)

# Never let git block on an interactive credential prompt; credentials come from the caller's
# configured helpers, SSH agent, or netrc.
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0"}
_GIT_TIMEOUT_SECONDS = 600
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
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return UNAUTHENTICATED
    if any(marker in lowered for marker in _AVAILABILITY_MARKERS):
        return TEMPORARILY_UNAVAILABLE
    return RESOURCE_DOES_NOT_EXIST


def _force_remove(func, path, _exc) -> None:
    # Git writes pack and object files read-only; on Windows that makes unlink fail.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_git_dir(dest: Path) -> None:
    git_dir = dest / ".git"
    if not git_dir.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(git_dir, onexc=_force_remove)
    else:
        shutil.rmtree(git_dir, onerror=_force_remove)
    if git_dir.exists():
        raise invalid_content(
            f"Could not remove the '.git' directory from the checkout at '{dest}'."
        )


def fetch_git(
    url: str, ref: str | None, dest: Path, *, max_bytes: int, subpath: str | None = None
) -> Path:
    """
    Materialize the tree at ``ref`` of the repository ``url`` into ``dest``.

    A shallow fetch of the single ref (or the remote ``HEAD`` when ``ref`` is omitted) keeps
    transfer small. The ``.git`` directory is removed so only content remains and the digest
    never sees repository internals. Submodules are not initialized; a skill is a plain
    content tree. The size limit applies to the tree at ``subpath``.
    """
    try:
        # GitPython needs the git executable at import time, so import only when fetching.
        import git
    except ImportError as e:
        raise MlflowException(
            "Fetching Git sources requires GitPython and a git executable on PATH. "
            f"Install git and `pip install gitpython`. Original error: {e}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(dest)
    try:
        with repo.git.custom_environment(**_GIT_ENV):
            origin = repo.create_remote("origin", url)
            origin.fetch(refspec=ref or "HEAD", depth=1, kill_after_timeout=_GIT_TIMEOUT_SECONDS)
            repo.git.checkout("FETCH_HEAD")
    except git.exc.GitCommandError as e:
        detail = (e.stderr or str(e)).strip()
        target = f"{url} at ref '{ref}'" if ref else url
        raise source_unavailable(target, detail, error_code=_error_code_for_git(detail))
    finally:
        repo.close()
    _remove_git_dir(dest)
    root = resolve_contained(dest, subpath)
    if (size := tree_size(root)) > max_bytes:
        raise invalid_content(
            f"Git checkout is {size} bytes, which exceeds the skill content size limit of "
            f"{max_bytes} bytes."
        )
    return dest
