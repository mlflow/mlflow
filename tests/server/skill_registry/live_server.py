import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.server import (
    ARTIFACTS_DESTINATION_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
    handlers,
    workspace_helpers,
)
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@contextlib.contextmanager
def serve_tracking_app(
    db_uri: str,
    artifacts_destination: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspaces: bool = False,
) -> Iterator[str]:
    """
    Serve the full MLflow app on a real port from a background thread, the way
    ``mlflow server --serve-artifacts [--enable-workspaces]`` does, and yield its URL.

    For tests whose client side must run for real: the SDK's ``requests`` transport, the CLI,
    and pull's artifact download. The app is the one imported in this process, so the stores it
    caches at module level are reset here and pointed at ``db_uri``.
    """
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(artifacts_destination))
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, str(workspaces).lower())
    monkeypatch.setattr(handlers, "_tracking_store", None)
    monkeypatch.setattr(handlers, "_model_registry_store", None)
    monkeypatch.setattr(handlers, "_artifact_repo", None)
    monkeypatch.setattr(workspace_helpers, "_workspace_store", None)
    initialize_backend_stores(
        db_uri,
        default_artifact_root="mlflow-artifacts:/",
        workspace_store_uri=db_uri if workspaces else None,
    )
    with ServerThread(app, get_safe_port()) as url:
        yield url
