import logging
import os
from functools import lru_cache

from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    TRACE_RESOLVE_TAGS_ALLOWLIST,
)

_logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def resolve_env_metadata():
    """
    Resolve common environment metadata to be saved in the trace info. These should not
    # change over time, so we resolve them only once. These will be stored in trace
    # metadata rather than tags, because they are immutable.
    """
    # GitRunContext does not property work in notebook because _get_main_file()
    # points to the kernel launcher file, not the actual notebook file.
    metadata = resolve_tags(ignore=[GitRunContext])
    if not is_in_databricks_notebook():
        # Get Git metadata for the script or notebook. If the notebook is in a
        # Databricks managed Git repo, DatabricksRepoRunContext the metadata
        # so we don't need to run this logic.
        metadata.update(_resolve_git_metadata())

    return {key: value for key, value in metadata.items() if key in TRACE_RESOLVE_TAGS_ALLOWLIST}


def _resolve_git_metadata():
    try:
        import git  # noqa: F401
    except ImportError:
        _logger.debug("Git python package is not installed. Skipping git metadata resolution.")
        return {}

    try:
        repo = os.getcwd()
        return {
            MLFLOW_GIT_COMMIT: get_git_commit(repo) or "",
            MLFLOW_GIT_REPO_URL: get_git_repo_url(repo) or "",
            MLFLOW_GIT_BRANCH: get_git_branch(repo) or "",
        }
    except Exception:
        _logger.debug("Failed to resolve git metadata", exc_info=True)

    return {}
