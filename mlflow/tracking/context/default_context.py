import sys
import getpass

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
)


_DEFAULT_USER = "unknown"


def _get_user():
    """Get the current computer username."""
    try:
        return getpass.getuser()
    except ImportError:
        return _DEFAULT_USER


def _get_main_file():
    if len(sys.argv) > 0:
        return sys.argv[0]
    return None


def _get_source_name():
    main_file = _get_main_file()
    if main_file is not None:
        return main_file
    return "<console>"


def _get_source_type():
    return SourceType.LOCAL


class DefaultRunContext(RunContextProvider):
    def in_context(self):
        return True

    def tags(self):
        return {
            MLFLOW_USER: _get_user(),
            MLFLOW_SOURCE_NAME: _get_source_name(),
            MLFLOW_SOURCE_TYPE: SourceType.to_string(_get_source_type()),
        }
