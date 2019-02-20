import os
import sys
import logging
from abc import ABCMeta, abstractmethod

from mlflow.entities import SourceType
from mlflow.utils.databricks_utils import (
    is_in_databricks_notebook, get_notebook_id, get_notebook_path,
    get_webapp_url
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_GIT_COMMIT,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_NOTEBOOK_ID
)


_logger = logging.getLogger(__name__)


def _get_main_file():
    if len(sys.argv) > 0:
        return sys.argv[0]
    return None


def _get_source_name():
    main_file = _get_main_file()
    if main_file is not None:
        return main_file
    return "<console>"


def _get_git_commit(path):
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound, NoSuchPathError
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s", e)
        return None
    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return None


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return _get_git_commit(main_file)
    return None


def _get_source_type():
    return SourceType.LOCAL


class ContextProvider:

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """
        Determine if MLflow is running in this context.

        :return: bool indicating if in this context
        """
        pass

    @abstractmethod
    def tags(self):
        """
        Generate context-specific tags.

        :return: dict of tags
        """
        return {}


class DefaultContext(ContextProvider):
    def in_context(self):
        return True

    def tags(self):
        return {
            MLFLOW_SOURCE_NAME: _get_source_name(),
            MLFLOW_SOURCE_TYPE: _get_source_type()
        }


class GitContext(ContextProvider):
    def in_context(self):
        return _get_source_version() is not None

    def tags(self):
        return {
            MLFLOW_GIT_COMMIT: _get_source_version()
        }


class DatabricksNotebookContext(ContextProvider):
    def in_context(self):
        return is_in_databricks_notebook()

    def tags(self):
        notebook_id = get_notebook_id()
        notebook_path = get_notebook_path()
        webapp_url = get_webapp_url()
        tags = {
            MLFLOW_SOURCE_NAME: notebook_path,
            MLFLOW_SOURCE_TYPE: SourceType.NOTEBOOK
        }
        if notebook_id is not None:
            tags[MLFLOW_DATABRICKS_NOTEBOOK_ID] = notebook_id
        if notebook_path is not None:
            tags[MLFLOW_DATABRICKS_NOTEBOOK_PATH] = notebook_path
        if webapp_url is not None:
            tags[MLFLOW_DATABRICKS_WEBAPP_URL] = webapp_url
        return tags
