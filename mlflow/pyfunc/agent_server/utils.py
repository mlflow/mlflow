import logging
import os
import subprocess
from contextvars import ContextVar

import mlflow
from mlflow.tracking.fluent import _set_active_model

# Context-isolated storage for request headers
# ensuring thread-safe access across async execution contexts
_request_headers: ContextVar[dict[str, str]] = ContextVar[dict[str, str]](
    "request_headers", default={}
)

logger = logging.getLogger(__name__)


def set_request_headers(headers: dict[str, str]) -> None:
    """Set request headers in the current context (called by server)"""
    _request_headers.set(headers)


def get_request_headers() -> dict[str, str]:
    """Get all request headers from the current context"""
    return _request_headers.get()


def get_header(name: str, default: str | None = None) -> str | None:
    """Get a specific header value from the current context"""
    return get_request_headers().get(name, default)


def get_forwarded_access_token() -> str | None:
    """Get the x-forwarded-access-token from the current request context."""
    return get_header("x-forwarded-access-token")


def get_obo_workspace_client():
    """Get a workspace client with the token from the
    `x-forwarded-access-token` header for OBO authentication
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError("databricks-sdk is required to use OBO authentication")
    return WorkspaceClient(token=get_forwarded_access_token(), auth_type="pat")


def setup_mlflow() -> None:
    """Initialize MLflow tracking and set active model."""
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    assert experiment_id is not None, (
        "You must set MLFLOW_EXPERIMENT_ID in your environment to enable MLflow git-based logging "
        "and real time tracing. Refer to the README for more info."
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_id=experiment_id)

    # in a Databricks App, the app name is set in the environment variable DATABRICKS_APP_NAME
    # in local development, we use a fallback app name
    app_name = os.getenv("DATABRICKS_APP_NAME", "local")

    # TODO: migrate over to `enable_git_model_versioning` or merge functionality
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
