"""
Utilities for MLflow cli server config validation and resolving.

NOTE: these functions are intended to be used as utilities for the cli click-based interface.
Do not use for any other purpose as the potential Exceptions being raised will be misleading
for users.
"""

import click
from typing import Optional

from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_local_uri


def resolve_default_artifact_root(
    serve_artifacts: bool,
    default_artifact_root: str,
    backend_store_uri: Optional[str],
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
                "Option 'default-artifact-root' is required, when backend store is not "
                "local file based."
            )
            eprint(msg)
            raise click.UsageError(message=msg)
    return default_artifact_root


def _validate_artifacts_only_config(
    serve_artifacts: bool, artifacts_only: bool, backend_store_uri: str
) -> None:
    """
    This utility prevents a confusing configuration error for users wherein the ``list_artifacts``
    functionality will not resolve correctly if the backend_store_uri is to a remote location.

    Note: this raises a click Exception and should only be used in conjunction with the cli
    interface.
    """
    if serve_artifacts and artifacts_only and not is_local_uri(backend_store_uri):
        msg = (
            "The server configuration is set as '--artifacts-only` with a non-local "
            f"'--backend-store-uri' provided: `{backend_store_uri}`. When using proxied "
            "artifact mode, the '--backend-store-uri' must be a local path or left as default. "
            "To set artifact storage location, use the '--artifacts-destination` argument."
        )
        eprint(msg)
        raise click.UsageError(message=msg)
