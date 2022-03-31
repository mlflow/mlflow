import sys

import click

from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_local_uri


def resolve_default_artifact_root(
    serve_artifacts, default_artifact_root, backend_store_uri, resolve_to_local=False
):
    if serve_artifacts and not default_artifact_root:
        default_artifact_root = DEFAULT_ARTIFACTS_URI
    elif not serve_artifacts and not default_artifact_root:
        if is_local_uri(backend_store_uri):
            default_artifact_root = backend_store_uri
        elif resolve_to_local:
            default_artifact_root = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
        else:
            eprint(
                "Option 'default-artifact-root' is required, when backend store is not "
                "local file based."
            )
            sys.exit(1)
    return default_artifact_root


def _validate_artifacts_only_config(serve_artifacts, artifacts_only, backend_store_uri):
    """
    This utility prevents a confusing configuration error for users wherein the ``list_artifacts``
    functionality will not resolve correctly if the backend_store_uri is to a remote location.
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
