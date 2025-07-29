import subprocess
from dataclasses import dataclass

from typing_extensions import Self


@dataclass(kw_only=True)
class GitInfo:
    branch: str | None = None
    commit: str | None = None
    dirty: bool = False

    @classmethod
    def from_env(cls) -> Self | None:
        if not cls._is_git_available():
            return None

        if not cls._is_in_git_repo():
            return None

        return cls(
            branch=cls._get_current_branch(),
            commit=cls._get_current_commit(),
            dirty=cls._is_repo_dirty(),
        )

    @staticmethod
    def _is_git_available() -> bool:
        return subprocess.call(["git", "--version"], stderr=subprocess.DEVNULL) == 0

    @staticmethod
    def _is_in_git_repo() -> bool:
        return subprocess.call(["git", "rev-parse", "--git-dir"], stderr=subprocess.DEVNULL) == 0

    @staticmethod
    def _get_current_branch() -> str | None:
        try:
            stdout = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            branch = stdout.strip()
            return branch if branch != "HEAD" else None
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def _get_current_commit() -> str | None:
        try:
            stdout = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            )
            return stdout.strip()
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def _is_repo_dirty() -> bool:
        try:
            result = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            )
            return bool(result.strip())
        except subprocess.CalledProcessError:
            return False
