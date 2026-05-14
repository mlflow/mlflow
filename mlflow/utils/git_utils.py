import logging
import os
from urllib.parse import urlsplit, urlunsplit

_logger = logging.getLogger(__name__)


def _strip_credentials_from_url(url: str) -> str:
    """
    Strip any embedded userinfo (username/password) from a URL so it's safe to record as a tag.

    HTTP(S) and similar remotes can carry credentials in the form
    ``https://user:token@host/path``; MLflow records repo URLs as run tags that are stored and
    displayed to users, so we drop the userinfo before exposing them. URLs without a scheme
    (e.g. SSH-style ``git@github.com:org/repo.git``) or without userinfo are returned unchanged.
    """
    parsed = urlsplit(url)
    if not parsed.scheme or "@" not in parsed.netloc:
        return url
    _, _, host_port = parsed.netloc.rpartition("@")
    return urlunsplit((parsed.scheme, host_port, parsed.path, parsed.query, parsed.fragment))


def get_git_repo_url(path: str) -> str | None:
    """
    Obtains the url of the git repository associated with the specified path,
    returning ``None`` if the path does not correspond to a git repository. Any embedded
    credentials (e.g. tokens in HTTPS remotes) are stripped before the URL is returned.
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
        url = next((remote.url for remote in repo.remotes), None)
        return _strip_credentials_from_url(url) if url is not None else None
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
