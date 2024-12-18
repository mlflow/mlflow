import functools
from typing import Optional

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.git_utils import GitInfo


class GitRunContext(RunContextProvider):
    @functools.cached_property
    def _git_info(self) -> Optional[GitInfo]:
        if main_file := _get_main_file():
            return GitInfo.from_path(main_file)
        return None

    def in_context(self):
        return self._git_info is not None

    def tags(self) -> dict[str, str]:
        if self._git_info:
            return self._git_info.to_mlflow_tags()
        return {}
