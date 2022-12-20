import logging

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

_logger = logging.getLogger(__name__)


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return get_git_commit(main_file)
    return None


class GitRunContext(RunContextProvider):
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
        return {MLFLOW_GIT_COMMIT: self._source_version}
