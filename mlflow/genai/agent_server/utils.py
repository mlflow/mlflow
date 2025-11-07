import logging
import os
import subprocess
from contextvars import ContextVar

from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental

# Context-isolated storage for request headers
# ensuring thread-safe access across async execution contexts
_request_headers: ContextVar[dict[str, str]] = ContextVar[dict[str, str]](
    "request_headers", default={}
)

logger = logging.getLogger(__name__)


@experimental(version="3.6.0")
def set_request_headers(headers: dict[str, str]) -> None:
    """Set request headers in the current context (called by server)"""
    _request_headers.set(headers)


@experimental(version="3.6.0")
def get_request_headers() -> dict[str, str]:
    """Get all request headers from the current context"""
    return _request_headers.get()


@experimental(version="3.6.0")
def setup_mlflow_git_based_version_tracking() -> None:
    """Initialize MLflow tracking and set active model with git-based version tracking."""
    # in a Databricks App, the app name is set in the environment variable DATABRICKS_APP_NAME
    # in local development, we use a fallback app name
    app_name = os.getenv("DATABRICKS_APP_NAME", "local")

    # Get current git commit hash for versioning
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[:8]
        )
        version_identifier = f"git-{git_commit}"
    except subprocess.CalledProcessError:
        version_identifier = "no-git"
    logged_model_name = f"{app_name}-{version_identifier}"

    # Set the active model context
    active_model_info = _set_active_model(name=logged_model_name)
    logger.info(
        f"Active LoggedModel: '{active_model_info.name}', Model ID: '{active_model_info.model_id}'"
    )
