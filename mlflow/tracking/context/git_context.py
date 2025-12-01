import logging

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.git_utils import get_git_commit, get_git_branch, get_git_repo_url
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH, MLFLOW_GIT_REPO_URL

_logger = logging.getLogger(__name__)


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return get_git_commit(main_file)
    return None


def _get_git_branch():
    main_file = _get_main_file()
    if main_file is not None:
        return get_git_branch(main_file)
    return None


def _get_git_repo_url():
    main_file = _get_main_file()
    if main_file is not None:
        return get_git_repo_url(main_file)
    return None


class GitRunContext(RunContextProvider):
    def __init__(self):
        self._cache = {}

    @property
    def _source_version(self):
        if "source_version" not in self._cache:
            self._cache["source_version"] = _get_source_version()
        return self._cache["source_version"]

    @property
    def _git_branch(self):
        if "git_branch" not in self._cache:
            self._cache["git_branch"] = _get_git_branch()
        return self._cache["git_branch"]

    @property
    def _git_repo_url(self):
        if "git_repo_url" not in self._cache:
            self._cache["git_repo_url"] = _get_git_repo_url()
        return self._cache["git_repo_url"]

    def in_context(self):
        return self._source_version is not None

    def tags(self):
        tags = {}
        
        # Add Git commit SHA if available
        if self._source_version is not None:
            tags[MLFLOW_GIT_COMMIT] = self._source_version
        
        # Add Git branch name if available
        if self._git_branch is not None:
            tags[MLFLOW_GIT_BRANCH] = self._git_branch
        
        # Add Git repository URL if available
        if self._git_repo_url is not None:
            tags[MLFLOW_GIT_REPO_URL] = self._git_repo_url
        
        return tags
