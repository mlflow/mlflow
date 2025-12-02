from __future__ import annotations

import os

from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.utils.workspace_context import WorkspaceContext, clear_workspace
from mlflow.utils.workspace_utils import (
    DEFAULT_WORKSPACE_NAME,
    resolve_entity_workspace_name,
)


def teardown_function():
    # Ensure the ContextVar does not leak between tests
    clear_workspace()
    os.environ.pop(MLFLOW_WORKSPACE.name, None)


def test_resolve_entity_workspace_prefers_argument():
    assert resolve_entity_workspace_name("  team-arg  ") == "team-arg"


def test_resolve_entity_workspace_uses_context_var():
    with WorkspaceContext("ctx-workspace"):
        assert resolve_entity_workspace_name(None) == "ctx-workspace"


def test_resolve_entity_workspace_falls_back_to_env(monkeypatch):
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)
    with WorkspaceContext(None):
        pass
    monkeypatch.setenv(MLFLOW_WORKSPACE.name, "  env-workspace  ")
    assert resolve_entity_workspace_name(None) == "env-workspace"


def test_resolve_entity_workspace_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)
    with WorkspaceContext(None):
        pass
    assert resolve_entity_workspace_name(None) == DEFAULT_WORKSPACE_NAME
