import logging
import warnings

from typing_extensions import Self

import mlflow
from mlflow.genai.git_versioning.git_info import GitInfo, GitOperationError
from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class GitContext:
    def __init__(self, remote_name: str = "origin") -> None:
        try:
            self.info = GitInfo.from_env(remote_name=remote_name)
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
        filter_string = self.info.to_search_filter_string()
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
                # Update tags to ensure they're current (especially git diff)
                mlflow.set_logged_model_tags(model_id=model.model_id, tags=git_tags)
            case _:
                _logger.info(
                    "No existing model found with the current git information. "
                    "Creating a new model."
                )
                model = mlflow.initialize_logged_model(tags=git_tags)

        self.active_model = _set_active_model(model_id=model.model_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        disable_git_model_versioning()


# Global variable to track the active git context
_active_context: GitContext | None = None


@experimental(version="3.4.0")
def enable_git_model_versioning(remote_name: str = "origin") -> GitContext:
    """
    Enable git model versioning and set the active context.

    Args:
        remote_name: The name of the git remote to use. Defaults to "origin".

    Returns:
        A GitContext instance containing the git information and active model.
    """
    global _active_context
    context = GitContext(remote_name=remote_name)
    _active_context = context
    return context


@experimental(version="3.4.0")
def disable_git_model_versioning() -> None:
    """
    Disable git model versioning and reset the active context.
    """
    global _active_context
    _active_context = None
    mlflow.clear_active_model()


def _get_active_git_context() -> GitContext | None:
    """
    Get the currently active git context, if any.

    Returns:
        The active GitContext instance or None if no context is active.
    """
    return _active_context
