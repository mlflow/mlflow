from dataclasses import dataclass

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
        try:
            import git
        except ImportError as e:
            raise GitOperationError("GitPython is not installed") from e

        # Create repo object once and extract all info
        try:
            repo = git.Repo()
        except git.InvalidGitRepositoryError as e:
            raise GitOperationError(f"Not in a git repository: {e}") from e

        try:
            # Get branch info
            if repo.head.is_detached:
                raise GitOperationError("In detached HEAD state, no branch name available")
            branch = repo.active_branch.name

            # Get commit info
            commit = repo.head.commit.hexsha

            # Check if repo is dirty
            dirty = repo.is_dirty(untracked_files=False)

            return cls(branch=branch, commit=commit, dirty=dirty)

        except git.GitError as e:
            raise GitOperationError(f"Failed to get repository information: {e}") from e
