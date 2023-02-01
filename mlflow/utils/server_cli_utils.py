"""
Utilities for MLflow cli server config validation and resolving.
NOTE: these functions are intended to be used as utilities for the cli click-based interface.
Do not use for any other purpose as the potential Exceptions being raised will be misleading
for users.
"""

import click

from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_local_uri


def resolve_default_artifact_root(
    serve_artifacts: bool,
    default_artifact_root: str,
    backend_store_uri: str,
    resolve_to_local: bool = False,
) -> str:
    if serve_artifacts and not default_artifact_root:
        default_artifact_root = DEFAULT_ARTIFACTS_URI
    elif not serve_artifacts and not default_artifact_root:
        if is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        elif resolve_to_local:
            default_artifact_root = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
        else:
            msg = (
                "Option 'default-artifact-root' is required when backend store is not "
                "local file based and artifact serving is disabled."
            )
            eprint(msg)
            raise click.UsageError(message=msg)
    return default_artifact_root


def _is_default_backend_store_uri(backend_store_uri: str) -> bool:
    """
    Utility function to validate if the configured backend store uri location is set as the
    default value for MLflow server.

    :param backend_store_uri: The value set for the backend store uri for MLflow server artifact
           handling.
    :return: bool True if the default value is set.
    """
    return backend_store_uri == DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH


def artifacts_only_config_validation(artifacts_only: bool, backend_store_uri: str) -> None:
    if artifacts_only and not _is_default_backend_store_uri(backend_store_uri):
        msg = (
            "You are starting a tracking server in `--artifacts-only` mode and have provided a "
            f"value for `--backend_store_uri`: '{backend_store_uri}'. A tracking server in "
            "`--artifacts-only` mode cannot have a value set for `--backend_store_uri` to "
            "properly proxy access to the artifact storage location."
        )
        raise click.UsageError(message=msg)
