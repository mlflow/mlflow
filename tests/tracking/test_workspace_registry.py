from __future__ import annotations

import pytest

from mlflow.store.workspace.rest_store import RestWorkspaceStore
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._workspace.registry import (
    UnsupportedWorkspaceStoreURIException,
    _get_workspace_store_registry,
    get_workspace_store,
)


@pytest.fixture(autouse=True)
def _clear_workspace_store_cache():
    registry = _get_workspace_store_registry()
    registry._get_store_with_resolved_uri.cache_clear()
    yield
    registry._get_store_with_resolved_uri.cache_clear()


def test_get_workspace_store_resolves_sqlalchemy(tmp_path):
    workspace_uri = f"sqlite:///{tmp_path / 'workspace.db'}"
    store = get_workspace_store(workspace_uri=workspace_uri)
    assert isinstance(store, SqlAlchemyStore)
    assert store._workspace_uri == workspace_uri
    store._engine.dispose()


def test_get_workspace_store_resolves_rest():
    store = get_workspace_store(workspace_uri="http://example.com")
    assert isinstance(store, RestWorkspaceStore)


def test_get_workspace_store_unsupported_scheme():
    with pytest.raises(
        UnsupportedWorkspaceStoreURIException,
        match="got unsupported URI 'foo://workspace'",
    ):
        get_workspace_store(workspace_uri="foo://workspace")
