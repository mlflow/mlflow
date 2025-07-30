from dataclasses import dataclass

import git
from typing_extensions import Self


class GitOperationError(Exception):
    """Raised when a git operation fails"""


@dataclass(kw_only=True)
class GitInfo:
    branch: str
    commit: str
    dirty: bool = False

    @classmethod
    def from_env(cls) -> Self:
        cls._is_git_available()
        cls._is_in_git_repo()

        return cls(
            branch=cls._get_current_branch(),
            commit=cls._get_current_commit(),
            dirty=cls._is_repo_dirty(),
        )

    @staticmethod
    def _is_git_available() -> bool:
        try:
            git.cmd.Git().version()
            return True
        except git.GitCommandError as e:
            raise GitOperationError(f"Git is not available or not installed: {e}") from e

    @staticmethod
    def _is_in_git_repo() -> bool:
        try:
            git.Repo()
            return True
        except git.InvalidGitRepositoryError as e:
            raise GitOperationError(f"Not in a git repository: {e}") from e

    @staticmethod
    def _get_current_branch() -> str:
        try:
            repo = git.Repo()
            if repo.head.is_detached:
                raise GitOperationError("In detached HEAD state, no branch name available")
            return repo.active_branch.name
        except git.GitError as e:
            raise GitOperationError(f"Failed to get current branch: {e}") from e

    @staticmethod
    def _get_current_commit() -> str:
        try:
            repo = git.Repo()
            return repo.head.commit.hexsha
        except git.GitError as e:
            raise GitOperationError(f"Failed to get current commit: {e}") from e

    @staticmethod
    def _is_repo_dirty() -> bool:
        try:
            repo = git.Repo()
            return repo.is_dirty(untracked_files=False)
        except git.GitError as e:
            raise GitOperationError(f"Failed to check repository status: {e}") from e
