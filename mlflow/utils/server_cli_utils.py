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
                "local file based."
            )
            eprint(msg)
            raise click.UsageError(message=msg)
    return default_artifact_root


def artifacts_only_config_warning(artifacts_only: bool, backend_store_uri: str) -> None:

    if artifacts_only and not is_local_uri(backend_store_uri):
        msg = (
            "You are starting a tracking server in `--artifacts-only` mode with a non-default "
            f"`--backend_store_uri`: '{backend_store_uri}'. To prevent errors in listing "
            "artifacts, please ensure that any other tracking servers that are started do not "
            "specify the argument `--backend_store_uri` at initialization."
        )
        click.echo(message=msg, nl=True, color=True)
