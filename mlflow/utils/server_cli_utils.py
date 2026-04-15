"""
Utilities for MLflow cli server config validation and resolving.
NOTE: these functions are intended to be used as utilities for the cli click-based interface.
Do not use for any other purpose as the potential Exceptions being raised will be misleading
for users.
"""

import click

from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.exceptions import MlflowException
from mlflow.store.tracking import (
    DEFAULT_ARTIFACTS_URI,
    DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    DEFAULT_TRACKING_URI,
)
from mlflow.utils.uri import is_local_uri


def assert_server_workspace_env_unset() -> None:
    """
    Ensure the server is not started with ``MLFLOW_WORKSPACE`` set.

    ``MLFLOW_WORKSPACE`` is a client-only setting used for propagating workspace across threads.
    Server isolation relies on per-request ContextVars and must not fall back to the env var.
    """
    if workspace := MLFLOW_WORKSPACE.get_raw():
        raise MlflowException.invalid_parameter_value(
            f"{MLFLOW_WORKSPACE.name}={workspace} is client-only. Unset it before starting the "
            + "server."
        )


def resolve_default_artifact_root(
    serve_artifacts: bool,
    default_artifact_root: str,
    backend_store_uri: str,
) -> str:
    if serve_artifacts and not default_artifact_root:
        default_artifact_root = DEFAULT_ARTIFACTS_URI
    elif not serve_artifacts and not default_artifact_root:
        if is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        else:
            default_artifact_root = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    return default_artifact_root


def _is_default_backend_store_uri(backend_store_uri: str | None) -> bool:
    """Utility function to validate if the configured backend store uri location is set as the
    default value for MLflow server.

    Args:
        backend_store_uri: The value set for the backend store uri for MLflow server artifact
            handling.

    Returns:
        bool True if the default value is set.

    """
    if backend_store_uri is None:
        return False
    return backend_store_uri in {DEFAULT_TRACKING_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH}


def artifacts_only_config_validation(
    artifacts_only: bool,
    backend_store_uri: str,
    enable_workspaces: bool = False,
) -> None:
    if artifacts_only and enable_workspaces:
        # Workspace mode relies on a workspace provider to resolve the default workspace and seed
        # request context. Artifact-only servers never load that stack, so they cannot determine
        # the active workspace safely. This decision can be revisited in the future.
        raise click.UsageError("--enable-workspaces cannot be combined with --artifacts-only.")
    if artifacts_only and not _is_default_backend_store_uri(backend_store_uri):
        msg = (
            "You are starting a tracking server in `--artifacts-only` mode and have provided a "
            f"value for `--backend_store_uri`: '{backend_store_uri}'. A tracking server in "
            "`--artifacts-only` mode cannot have a value set for `--backend_store_uri` to "
            "properly proxy access to the artifact storage location."
        )
        raise click.UsageError(message=msg)
