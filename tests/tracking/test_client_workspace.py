from __future__ import annotations

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE_STORE_URI
from mlflow.tracking._workspace import fluent as workspace_fluent
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


def test_mlflow_client_resolves_workspace_uri(monkeypatch):
    recorded: dict[str, str | None] = {}

    class DummyTrackingClient:
        def __init__(self, tracking_uri: str):
            recorded["tracking_uri"] = tracking_uri
            self.tracking_uri = tracking_uri

    class DummyWorkspaceClient:
        def __init__(self, workspace_uri: str | None = None):
            recorded["workspace_uri"] = workspace_uri

        # Methods invoked downstream are irrelevant for this initialization test.
        def list_workspaces(self):
            return []

    monkeypatch.setattr("mlflow.tracking.client.TrackingServiceClient", DummyTrackingClient)
    monkeypatch.setattr("mlflow.tracking.client.TracingClient", lambda _: None)
    monkeypatch.setattr("mlflow.tracking.client.WorkspaceProviderClient", DummyWorkspaceClient)
    monkeypatch.setattr(
        "mlflow.tracking.client.utils._resolve_tracking_uri",
        lambda uri: uri or "sqlite:///tracking.db",
    )
    monkeypatch.setattr(
        "mlflow.tracking.client.registry_utils._resolve_registry_uri",
        lambda registry_uri, tracking_uri: "registry-resolved",
    )

    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "sqlite:///tracking.db")
    monkeypatch.setenv(MLFLOW_WORKSPACE_STORE_URI.name, "sqlite:///workspace.db")

    client = MlflowClient()
    client.list_workspaces()
    assert recorded["workspace_uri"] == "sqlite:///workspace.db"

    recorded.clear()
    client = MlflowClient(workspace_store_uri="sqlite:///explicit.db")
    client.list_workspaces()
    assert recorded["workspace_uri"] == "sqlite:///explicit.db"

    recorded.clear()
    monkeypatch.delenv(MLFLOW_WORKSPACE_STORE_URI.name, raising=False)
    client = MlflowClient()
    client.list_workspaces()
    # Falls back to the tracking URI when workspace URI is unset.
    assert recorded["workspace_uri"] == "sqlite:///tracking.db"


def test_set_workspace_sets_env_and_context(monkeypatch):
    calls: dict[str, list[str]] = {"set_context_workspace": []}
    env: dict[str, str | None] = {}

    monkeypatch.setattr(
        workspace_fluent,
        "WorkspaceNameValidator",
        type(
            "Validator",
            (),
            {"validate": lambda name: calls.setdefault("validate", []).append(name)},
        ),
    )
    monkeypatch.setattr(
        workspace_fluent,
        "set_context_workspace",
        lambda name: (
            calls["set_context_workspace"].append(name),
            env.__setitem__("value", name),
        ),
    )

    workspace_fluent.set_workspace("team-space")

    assert calls["validate"] == ["team-space"]
    assert calls["set_context_workspace"] == ["team-space"]
    assert env["value"] == "team-space"


def test_set_workspace_clears_when_none(monkeypatch):
    env: dict[str, str | None] = {}
    calls = {"clear_workspace": 0, "set_context_workspace": [], "validate": []}

    monkeypatch.setattr(
        workspace_fluent,
        "WorkspaceNameValidator",
        type(
            "Validator",
            (),
            {"validate": lambda name: calls["validate"].append(name)},
        ),
    )
    monkeypatch.setattr(
        workspace_fluent,
        "set_context_workspace",
        lambda name: (
            calls["set_context_workspace"].append(name),
            env.__setitem__("value", name),
            calls.__setitem__(
                "clear_workspace", calls["clear_workspace"] + (1 if name is None else 0)
            ),
        ),
    )
    # Ensure default workspace does not trigger validation but does set env
    workspace_fluent.set_workspace(DEFAULT_WORKSPACE_NAME)
    assert calls["validate"] == []
    assert env["value"] == DEFAULT_WORKSPACE_NAME

    workspace_fluent.set_workspace(None)
    assert calls["clear_workspace"] == 1
    assert env.get("value") is None
