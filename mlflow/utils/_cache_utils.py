from huggingface_hub import scan_cache_dir


def _clear_hub_cache():
    """
    Frees up disk space for cached huggingface transformers models and components.

    This function will remove all files within the cache if the total size of objects exceeds
    1 GiB on disk. It is used only in CI testing to alleviate the disk burden on the runners as
    they have limited allocated space and will terminate if the available disk space drops too low.
    """
    full_cache = scan_cache_dir()
    cache_size_in_gb = full_cache.size_on_disk / 1000**3

    if cache_size_in_gb > 1:
        commits_to_purge = [rev.commit_hash for repo in full_cache.repos for rev in repo.revisions]
        delete_strategy = full_cache.delete_revisions(*commits_to_purge)
        delete_strategy.execute()
