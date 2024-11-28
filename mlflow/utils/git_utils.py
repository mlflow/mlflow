import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlflow.utils import mlflow_tags

_logger = logging.getLogger(__name__)


@dataclass
class GitInfo:
    commit: str
    branch: str
    url: Optional[str] = None

    def to_mlflow_tags(self) -> dict[str, str]:
        tags = {
            mlflow_tags.MLFLOW_GIT_COMMIT: self.commit,
            mlflow_tags.LEGACY_MLFLOW_GIT_BRANCH_NAME: self.branch,
            mlflow_tags.MLFLOW_GIT_BRANCH: self.branch,
        }
        if self.url:
            tags[mlflow_tags.MLFLOW_GIT_REPO_URL] = self.url
            tags[mlflow_tags.LEGACY_MLFLOW_GIT_REPO_URL] = self.url
        return tags

    @classmethod
    def load(cls, path: str) -> Optional["GitInfo"]:
        try:
            from git import Repo
        except ImportError as e:
            _logger.warning(
                "Failed to import Git (the Git executable is probably not on your PATH),"
                " so Git SHA is not available. Error: %s",
                e,
            )
            return None

        try:
            path = Path(path)
            if path.is_file():
                path = path.parent
            repo = Repo(path, search_parent_directories=True)
            if str(path) in repo.ignored(path):
                return None

            return cls(
                commit=repo.head.commit.hexsha,
                branch=repo.active_branch.name,
                url=next((r.url for r in repo.remotes), None),
            )
        except Exception:
            return None


def get_git_repo_url(path: str) -> Optional[str]:
    """
    Obtains the url of the git repository associated with the specified path,
    returning ``None`` if the path does not correspond to a git repository.
    """
    return info.url if (info := GitInfo.load(path)) else None


def get_git_commit(path: str) -> Optional[str]:
    """
    Obtains the hash of the latest commit on the current branch of the git repository associated
    with the specified path, returning ``None`` if the path does not correspond to a git
    repository.
    """
    return info.commit if (info := GitInfo.load(path)) else None


def get_git_branch(path: str) -> Optional[str]:
    """
    Obtains the name of the current branch of the git repository associated with the specified
    path, returning ``None`` if the path does not correspond to a git repository.
    """
    return info.branch if (info := GitInfo.load(path)) else None
