from __future__ import annotations

import shutil
from pathlib import Path

from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import tree_size

# Never let git block on an interactive credential prompt; credentials come from the caller's
# configured helpers, SSH agent, or netrc.
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0"}


def fetch_git(url: str, ref: str | None, dest: Path, *, max_bytes: int) -> Path:
    """
    Materialize the tree at ``ref`` of the repository ``url`` into ``dest``.

    A shallow fetch of the single ref keeps transfer small. When ``ref`` is omitted the remote
    HEAD is used. The ``.git`` directory is removed so only content remains. Submodules are
    not initialized; a skill is a plain content tree.
    """
    # GitPython requires the git executable at import time, so import only when fetching.
    import git

    dest.mkdir(parents=True, exist_ok=True)
    try:
        if ref is None:
            git.Repo.clone_from(url, dest, depth=1, env=_GIT_ENV)
        else:
            repo = git.Repo.init(dest)
            with repo.git.custom_environment(**_GIT_ENV):
                origin = repo.create_remote("origin", url)
                origin.fetch(refspec=ref, depth=1)
                repo.git.checkout("FETCH_HEAD")
    except git.exc.GitCommandError as e:
        detail = e.stderr or str(e)
        target = f"{url} at ref '{ref}'" if ref else url
        raise source_unavailable(target, detail.strip())
    shutil.rmtree(dest / ".git", ignore_errors=True)
    if (size := tree_size(dest)) > max_bytes:
        raise invalid_content(
            f"Git checkout is {size} bytes, which exceeds the skill content size limit of "
            f"{max_bytes} bytes."
        )
    return dest
