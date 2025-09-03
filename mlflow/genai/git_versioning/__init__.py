import logging

from typing_extensions import Self

import mlflow
from mlflow.genai.git_versioning.git_info import GitInfo, GitOperationError
from mlflow.telemetry.events import GitModelVersioningEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class GitContext:
    def __init__(self, remote_name: str = "origin") -> None:
        try:
            self.info = GitInfo.from_env(remote_name=remote_name)
        except GitOperationError as e:
            _logger.warning(
                f"Encountered an error while retrieving git information: {e}. "
                f"Git model versioning is disabled."
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


@record_usage_event(GitModelVersioningEvent)
def _enable_git_model_versioning(remote_name: str) -> None:
    global _active_context
    context = GitContext(remote_name=remote_name)
    _active_context = context
    return context


@experimental(version="3.4.0")
def enable_git_model_versioning(remote_name: str = "origin") -> GitContext:
    """
    Enable automatic Git-based model versioning for MLflow traces.

    This function enables automatic version tracking based on your Git repository state.
    When enabled, MLflow will:
    - Detect the current Git branch, commit hash, and dirty state
    - Create or reuse a LoggedModel matching this exact Git state
    - Link all subsequent traces to this LoggedModel version
    - Capture uncommitted changes as diffs when the repository is dirty

    Args:
        remote_name: The name of the git remote to use for repository URL detection.
                    Defaults to "origin".

    Returns:
        A GitContext instance containing:
        - info: GitInfo object with branch, commit, dirty state, and diff information
        - active_model: The active LoggedModel linked to current Git state

    Example:

    .. code-block:: python

        import mlflow.genai

        # Enable Git-based versioning
        context = mlflow.genai.enable_git_model_versioning()
        print(f"Branch: {context.info.branch}, Commit: {context.info.commit[:8]}")
        # Output: Branch: main, Commit: abc12345


        # All traces are now automatically linked to this Git version
        @mlflow.trace
        def my_app():
            return "result"


        # Can also use as a context manager
        with mlflow.genai.enable_git_model_versioning() as context:
            # Traces within this block are linked to the Git version
            result = my_app()

    Note:
        If Git is not available or the current directory is not a Git repository,
        a warning is issued and versioning is disabled (context.info will be None).
    """
    return _enable_git_model_versioning(remote_name)


@experimental(version="3.4.0")
def disable_git_model_versioning() -> None:
    """
    Disable Git-based model versioning and clear the active model context.

    This function stops automatic Git-based version tracking and clears any active
    LoggedModel context. After calling this, traces will no longer be automatically
    linked to Git-based versions.

    This is automatically called when exiting a context manager created with
    enable_git_model_versioning().

    Example:

    .. code-block:: python

        import mlflow.genai

        # Enable versioning
        context = mlflow.genai.enable_git_model_versioning()
        # ... do work with versioning enabled ...

        # Disable versioning
        mlflow.genai.disable_git_model_versioning()
        # Traces are no longer linked to Git versions
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
