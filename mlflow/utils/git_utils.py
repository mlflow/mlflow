from typing import Optional


def get_git_repo_url(path: str) -> Optional[str]:
    """
    Obtains the url of the git repository associated with the specified path,
    returning ``None`` if the path does not correspond to a git repository.
    """
    from git import Repo

    try:
        repo = Repo(path, search_parent_directories=True)
        remote_urls = [remote.url for remote in repo.remotes]
        if len(remote_urls) == 0:
            return None
    except Exception:
        return None
    return remote_urls[0]
