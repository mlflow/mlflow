import logging
import warnings

from typing_extensions import Self

import mlflow
from mlflow.genai.git_versioning.git_info import GitInfo, GitOperationError
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class GitContext:
    def __init__(self) -> None:
        try:
            self.info = GitInfo.from_env()
        except GitOperationError as e:
            warnings.warn(
                (
                    f"Encountered an error while retrieving git information: {e}. "
                    f"Git model versioning is disabled."
                ),
                UserWarning,
                stacklevel=2,
            )
            self.info = None
            self.active_model = None
            return

        git_tags = self.info.to_mlflow_tags()
        filter_string = " AND ".join(f"tags.`{k}` = '{v}'" for k, v in git_tags.items())
        models = mlflow.search_logged_models(
            filter_string=filter_string,
            max_results=1,
            output_format="list",
        )
        match models:
            case [m]:
                _logger.info(
                    f"Using existing model with branch '{self.info.branch}', "
                    f"commit '{self.info.commit}', dirty state '{self.info.dirty}'."
                )
                model = m
            case _:
                _logger.info(
                    "No existing model found with the current git information. "
                    "Creating a new model."
                )
                model = mlflow.initialize_logged_model(tags=git_tags)

        self.active_model = mlflow._set_active_model(model_id=model.model_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        disable_git_model_versioning()


# Global variable to track the active git context
_active_context: GitContext | None = None


@experimental(version="3.3.0")
def enable_git_model_versioning() -> GitContext:
    """
    Enable git model versioning and set the active context.
    """
    global _active_context
    context = GitContext()
    _active_context = context
    return context


@experimental(version="3.3.0")
def disable_git_model_versioning() -> None:
    """
    Disable git model versioning and reset the active context.
    """
    global _active_context
    mlflow.clear_active_model()
    _active_context = None


def _get_active_git_context() -> GitContext | None:
    """
    Get the currently active git context, if any.

    Returns:
        The active GitContext instance or None if no context is active.
    """
    return _active_context
