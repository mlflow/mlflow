import logging
import os

_logger = logging.getLogger(__name__)


def get_git_repo_url(path: str) -> str | None:
    """
    Obtains the url of the git repository associated with the specified path,
    returning ``None`` if the path does not correspond to a git repository.
    """
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
        repo = Repo(path, search_parent_directories=True)
        return next((remote.url for remote in repo.remotes), None)
    except Exception:
        return None


def get_git_commit(path: str) -> str | None:
    """
    Obtains the hash of the latest commit on the current branch of the git repository associated
    with the specified path, returning ``None`` if the path does not correspond to a git
    repository.
    """
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
        if os.path.isfile(path):
            path = os.path.dirname(os.path.abspath(path))
        repo = Repo(path, search_parent_directories=True)
        if path in repo.ignored(path):
            return None
        return repo.head.commit.hexsha
    except Exception:
        return None


def get_git_branch(path: str) -> str | None:
    """
    Obtains the name of the current branch of the git repository associated with the specified
    path, returning ``None`` if the path does not correspond to a git repository.
    """
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
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = Repo(path, search_parent_directories=True)
        return repo.active_branch.name
    except Exception:
        return None
