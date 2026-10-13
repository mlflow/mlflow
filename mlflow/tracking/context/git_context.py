import logging

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
)

_logger = logging.getLogger(__name__)


def _resolve_git_info():
    main_file = _get_main_file()
    if main_file is None:
        return {}
    return {
        MLFLOW_GIT_COMMIT: get_git_commit(main_file),
        MLFLOW_GIT_BRANCH: get_git_branch(main_file),
        MLFLOW_GIT_REPO_URL: get_git_repo_url(main_file),
    }


class GitRunContext(RunContextProvider):
    def __init__(self):
        self._cache = {}

    @property
    def _git_info(self):
        if "git_info" not in self._cache:
            self._cache["git_info"] = _resolve_git_info()
        return self._cache["git_info"]

    def in_context(self):
        return self._git_info.get(MLFLOW_GIT_COMMIT) is not None

    def tags(self):
        return {k: v for k, v in self._git_info.items() if v is not None}
