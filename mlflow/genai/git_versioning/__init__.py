import warnings

from typing_extensions import Self

from mlflow.genai.git_versioning.git_info import GitInfo


class GitContext:
    def __init__(self) -> None:
        self.info = GitInfo.from_env()
        if self.info is None:
            warnings.warn(
                (
                    "Git information is not available. Make sure git is installed "
                    "and you are in a git repository."
                ),
                UserWarning,
                stacklevel=2,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        disable_git_model_versioning()


# Global variable to track the active git context
_active_context: GitContext | None = None


def enable_git_model_versioning() -> GitContext:
    """
    Enable git model versioning and set the active context.
    """
    global _active_context
    context = GitContext()
    _active_context = context
    return context


def disable_git_model_versioning() -> None:
    """
    Disable git model versioning and reset the active context.
    """
    global _active_context
    _active_context = None


def _get_active_context() -> GitContext | None:
    """
    Get the currently active git context, if any.

    Returns:
        The active GitContext instance or None if no context is active.
    """
    return _active_context
