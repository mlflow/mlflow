import pytest

from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.utils.workspace_utils import (
    DEFAULT_WORKSPACE_NAME,
    resolve_entity_workspace_name,
)


@pytest.fixture(autouse=True)
def _clear_workspace_env(monkeypatch):
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)


def test_resolve_entity_workspace_name_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv(MLFLOW_WORKSPACE.name, "env-ws")
    monkeypatch.setattr(
        "mlflow.tracking._workspace.context.get_current_workspace",
        lambda: "context-ws",
    )

    assert resolve_entity_workspace_name("explicit-ws") == "explicit-ws"


def test_resolve_entity_workspace_name_uses_context_workspace(monkeypatch):
    monkeypatch.setattr(
        "mlflow.tracking._workspace.context.get_current_workspace",
        lambda: "context-ws",
    )

    assert resolve_entity_workspace_name(None) == "context-ws"


def test_resolve_entity_workspace_name_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(
        "mlflow.tracking._workspace.context.get_current_workspace",
        lambda: None,
    )
    monkeypatch.setenv(MLFLOW_WORKSPACE.name, "env-ws")

    assert resolve_entity_workspace_name(None) == "env-ws"


def test_resolve_entity_workspace_name_defaults_when_unset(monkeypatch):
    monkeypatch.setattr(
        "mlflow.tracking._workspace.context.get_current_workspace",
        lambda: None,
    )

    assert resolve_entity_workspace_name(None) == DEFAULT_WORKSPACE_NAME
