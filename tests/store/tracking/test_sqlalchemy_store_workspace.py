from __future__ import annotations

from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_WORKSPACE_DISABLE_ARTIFACT_PREFIX
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_utils import WORKSPACES_DIR_NAME


ARTIFACT_ROOT = "s3://my-bucket/mlflow"
WORKSPACE = "my-workspace"


@pytest.fixture
def store(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with mock.patch.object(
        WorkspaceAwareSqlAlchemyStore,
        "_initialize_store_state",
    ):
        s = WorkspaceAwareSqlAlchemyStore(db_uri, ARTIFACT_ROOT)
    return s


def _mock_provider(root, should_append):
    provider = mock.MagicMock()
    provider.resolve_artifact_root.return_value = (root, should_append)
    return provider


def test_artifact_location_appends_workspace_prefix_by_default(store):
    store._workspace_provider = _mock_provider(ARTIFACT_ROOT, True)
    location = store._get_artifact_location(42, WORKSPACE)
    assert f"/{WORKSPACES_DIR_NAME}/{WORKSPACE}/" in location
    assert location.endswith("/42")


def test_artifact_location_no_prefix_when_provider_returns_false(store):
    store._workspace_provider = _mock_provider(ARTIFACT_ROOT, False)
    location = store._get_artifact_location(42, WORKSPACE)
    assert WORKSPACES_DIR_NAME not in location
    assert location.endswith("/42")


@pytest.mark.parametrize("experiment_id", [0, 42])
def test_artifact_location_disable_prefix_env_var(store, experiment_id):
    store._workspace_provider = _mock_provider(ARTIFACT_ROOT, True)
    with mock.patch.object(
        MLFLOW_WORKSPACE_DISABLE_ARTIFACT_PREFIX, "get", return_value=True
    ):
        location = store._get_artifact_location(experiment_id, WORKSPACE)
    assert WORKSPACES_DIR_NAME not in location
    assert location.endswith(f"/{experiment_id}")
