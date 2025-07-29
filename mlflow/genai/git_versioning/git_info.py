import subprocess
from dataclasses import dataclass

from typing_extensions import Self


class GitOperationError(Exception):
    """Raised when a git operation fails"""

    @classmethod
    def from_called_process_error(
        cls, operation: str, error: subprocess.CalledProcessError
    ) -> Self:
        """Create GitOperationError from CalledProcessError with stderr details"""
        message = f"{operation}: {error}"
        if error.stderr and error.stderr.strip():
            message += f" (stderr: {error.stderr.strip()})"
        return cls(message)


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
            subprocess.check_call(
                ["git", "--version"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            raise GitOperationError.from_called_process_error(
                "Git is not available or not installed", e
            ) from e

    @staticmethod
    def _is_in_git_repo() -> bool:
        try:
            subprocess.check_call(
                ["git", "rev-parse", "--git-dir"],
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError as e:
            raise GitOperationError.from_called_process_error("Not in a git repository", e) from e

    @staticmethod
    def _get_current_branch() -> str:
        try:
            stdout = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.PIPE,
                text=True,
            )
            branch = stdout.strip()
            if branch == "HEAD":
                raise GitOperationError("In detached HEAD state, no branch name available")
            return branch
        except subprocess.CalledProcessError as e:
            raise GitOperationError.from_called_process_error(
                "Failed to get current branch", e
            ) from e

    @staticmethod
    def _get_current_commit() -> str:
        try:
            stdout = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, text=True
            )
            return stdout.strip()
        except subprocess.CalledProcessError as e:
            raise GitOperationError.from_called_process_error(
                "Failed to get current commit", e
            ) from e

    @staticmethod
    def _is_repo_dirty() -> bool:
        try:
            result = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.PIPE, text=True
            )
            return bool(result.strip())
        except subprocess.CalledProcessError as e:
            raise GitOperationError.from_called_process_error(
                "Failed to check repository status", e
            ) from e
