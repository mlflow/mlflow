import logging
import os

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_DIRTY,
    MLFLOW_GIT_REPO_URL,
)

_logger = logging.getLogger(__name__)


def _get_main_file_path():
    return _get_main_file()


class GitRunContext(RunContextProvider):
    def __init__(self):
        self._cache = {}

    def _resolve(self):
        """Lazily resolve all git metadata using a single ``git.Repo`` instance.

        Results are cached so that repeated calls to :meth:`in_context` and
        :meth:`tags` do not create additional ``Repo`` objects.
        """
        if "resolved" in self._cache:
            return
        self._cache["resolved"] = True

        main_file = _get_main_file_path()
        if main_file is None:
            self._cache["source_version"] = None
            return

        try:
            from git import Repo
        except ImportError as e:
            _logger.warning(
                "Failed to import Git (the Git executable is probably not on your PATH),"
                " so Git metadata is not available. Error: %s",
                e,
            )
            self._cache["source_version"] = None
            return

        # Normalise *main_file* to a directory so every look-up uses a
        # consistent path (get_git_repo_url and others expect a directory).
        path = main_file
        if os.path.isfile(path):
            path = os.path.dirname(os.path.abspath(path))

        try:
            repo = Repo(path, search_parent_directories=True)
        except Exception:
            self._cache["source_version"] = None
            return

        # ── commit ──
        try:
            if path in repo.ignored(path):
                self._cache["source_version"] = None
                return
            commit = repo.head.commit.hexsha
        except Exception:
            self._cache["source_version"] = None
            return

        self._cache["source_version"] = commit

        # ── branch ──
        try:
            self._cache["branch"] = repo.active_branch.name
        except (TypeError, ValueError):
            # Detached HEAD – branch is unavailable
            pass

        # ── repo URL ──
        try:
            self._cache["repo_url"] = next(
                (remote.url for remote in repo.remotes), None
            )
        except Exception:
            pass

        # ── dirty state ──
        try:
            # Use untracked_files=False to align with
            # mlflow.genai.git_versioning.git_info.GitInfo.from_env
            self._cache["is_dirty"] = repo.is_dirty(untracked_files=False)
        except Exception:
            pass

    def in_context(self):
        self._resolve()
        return self._cache.get("source_version") is not None

    def tags(self):
        self._resolve()
        tags = {MLFLOW_GIT_COMMIT: self._cache["source_version"]}

        branch = self._cache.get("branch")
        if branch is not None:
            tags[MLFLOW_GIT_BRANCH] = branch

        repo_url = self._cache.get("repo_url")
        if repo_url is not None:
            tags[MLFLOW_GIT_REPO_URL] = repo_url

        is_dirty = self._cache.get("is_dirty")
        if is_dirty is not None:
            # Use lowercase ("true"/"false") to stay consistent with
            # mlflow.genai.git_versioning.git_info.GitInfo.to_mlflow_tags
            tags[MLFLOW_GIT_DIRTY] = "true" if is_dirty else "false"

        return tags
