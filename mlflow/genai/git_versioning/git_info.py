import logging
from dataclasses import dataclass

from typing_extensions import Self

from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_DIFF,
    MLFLOW_GIT_DIRTY,
    MLFLOW_GIT_REPO_URL,
)

_logger = logging.getLogger(__name__)


class GitOperationError(Exception):
    """Raised when a git operation fails"""


@dataclass(kw_only=True)
class GitInfo:
    branch: str
    commit: str
    dirty: bool = False
    repo_url: str | None = None
    diff: str | None = None

    @classmethod
    def from_env(cls, remote_name: str) -> Self:
        try:
            import git
        except ImportError as e:
            # GitPython throws `ImportError` if `git` is unavailable.
            raise GitOperationError(str(e))

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

            # Get git diff if dirty
            diff: str | None = None
            if dirty:
                # Get the diff of unstaged changes
                diff = repo.git.diff(cached=False)
                # Get staged changes
                if staged_diff := repo.git.diff(cached=True):
                    diff = (diff + "\n" + staged_diff) if diff else staged_diff

            # Get repository URL
            repo_url = next((r.url for r in repo.remotes if r.name == remote_name), None)
            if repo_url is None:
                _logger.warning(
                    f"No remote named '{remote_name}' found. Repository URL will not be set."
                )
            return cls(branch=branch, commit=commit, dirty=dirty, repo_url=repo_url, diff=diff)

        except git.GitError as e:
            raise GitOperationError(f"Failed to get repository information: {e}") from e

    def to_mlflow_tags(self) -> dict[str, str]:
        tags = {
            MLFLOW_GIT_BRANCH: self.branch,
            MLFLOW_GIT_COMMIT: self.commit,
            MLFLOW_GIT_DIRTY: str(self.dirty).lower(),
        }
        if self.repo_url is not None:
            tags[MLFLOW_GIT_REPO_URL] = self.repo_url
        if self.diff is not None:
            tags[MLFLOW_GIT_DIFF] = self.diff
        return tags

    def to_search_filter_string(self) -> str:
        """
        Generate a filter string for search_logged_models.
        Excludes MLFLOW_GIT_DIFF from the filter as it's not meant for searching.
        """
        tags = {
            MLFLOW_GIT_BRANCH: self.branch,
            MLFLOW_GIT_COMMIT: self.commit,
            MLFLOW_GIT_DIRTY: str(self.dirty).lower(),
        }
        if self.repo_url is not None:
            tags[MLFLOW_GIT_REPO_URL] = self.repo_url

        return " AND ".join(f"tags.`{k}` = '{v}'" for k, v in tags.items())
