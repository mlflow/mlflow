import logging
from functools import lru_cache

from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_repo_context import DatabricksRepoRunContext
from mlflow.tracking.context.default_context import _get_main_file
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
    # NB: Skip unused context to avoid extra overhead.
    if is_in_databricks_notebook():
        metadata = resolve_tags(ignore=[DatabricksRepoRunContext, GitRunContext])
    else:
        metadata = resolve_tags(
            ignore=[DatabricksNotebookRunContext, DatabricksRepoRunContext, GitRunContext]
        )
        metadata.update(_resolve_git_metadata())

    return {key: value for key, value in metadata.items() if key in TRACE_RESOLVE_TAGS_ALLOWLIST}


def _resolve_git_metadata():
    try:
        if main_file := _get_main_file():
            return {
                MLFLOW_GIT_COMMIT: get_git_commit(main_file) or "",
                MLFLOW_GIT_REPO_URL: get_git_repo_url(main_file) or "",
                MLFLOW_GIT_BRANCH: get_git_branch(main_file) or "",
            }
    except Exception:
        _logger.debug("Failed to resolve git metadata", exc_info=True)

    return {}
