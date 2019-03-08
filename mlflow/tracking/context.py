import os
import sys
import logging
import itertools
import warnings
from abc import ABCMeta, abstractmethod

import entrypoints
from six.moves import reduce

from mlflow.entities import SourceType
from mlflow.utils import databricks_utils
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
        import git
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s", e)
        return None
    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = git.Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (git.InvalidGitRepositoryError, git.GitCommandNotFound, ValueError, git.NoSuchPathError):
        return None


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return _get_git_commit(main_file)
    return None


def _get_source_type():
    return SourceType.LOCAL


class ContextProvider(object):

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
        pass


class DefaultContext(ContextProvider):
    def in_context(self):
        return True

    def tags(self):
        return {
            MLFLOW_SOURCE_NAME: _get_source_name(),
            MLFLOW_SOURCE_TYPE: _get_source_type()
        }


class GitContext(ContextProvider):

    def __init__(self):
        self._cache = {}

    @property
    def _source_version(self):
        if "source_version" not in self._cache:
            self._cache["source_version"] = _get_source_version()
        return self._cache["source_version"]

    def in_context(self):
        return self._source_version is not None

    def tags(self):
        return {
            MLFLOW_GIT_COMMIT: self._source_version
        }


class DatabricksNotebookContext(ContextProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_notebook()

    def tags(self):
        notebook_id = databricks_utils.get_notebook_id()
        notebook_path = databricks_utils.get_notebook_path()
        webapp_url = databricks_utils.get_webapp_url()
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


class ContextProviderRegistry(object):
    """Registry for context provider implementations

    This class allows the registration of a context provider which can be used to infer meta
    information about the context of an MLflow experiment run. Implementations declared though the
    entrypoints `mlflow.context_provider` group can be automatically registered through the
    `register_entrypoints` method.
    """

    def __init__(self):
        self._registry = []

    def register(self, context_provider_cls):
        self._registry.append(context_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all("mlflow.context_provider"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register context provider "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2
                )

    def __iter__(self):
        return iter(self._registry)


_context_provider_registry = ContextProviderRegistry()
_context_provider_registry.register(DefaultContext)
_context_provider_registry.register(GitContext)
_context_provider_registry.register(DatabricksNotebookContext)

_context_provider_registry.register_entrypoints()


def _merge_tags(base, new):
    return dict(itertools.chain(base.items(), new.items()))


def resolve_tags(tags=None):

    tag_sets = []
    for provider in _context_provider_registry:
        if provider.in_context():
            tag_sets.append(provider.tags())

    if tags is not None:
        tag_sets.append(tags)

    return reduce(_merge_tags, tag_sets)
