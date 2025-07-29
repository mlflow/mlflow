import warnings

from typing_extensions import Self

from mlflow.genai.git_versioning.git_info import GitInfo


class GitContext:
    def __init__(self) -> None:
        self.info: GitInfo | None = GitInfo.from_env()
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
        pass


def enable_git_model_versioning() -> GitContext:
    """
    TODO
    """
    return GitContext()


def disable_git_model_versioning() -> None:
    """
    TODO
    """
