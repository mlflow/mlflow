from __future__ import annotations

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE_URI


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
    monkeypatch.setenv(MLFLOW_WORKSPACE_URI.name, "sqlite:///workspace.db")

    client = MlflowClient()
    client.list_workspaces()
    assert recorded["workspace_uri"] == "sqlite:///workspace.db"

    recorded.clear()
    client = MlflowClient(workspace_uri="sqlite:///explicit.db")
    client.list_workspaces()
    assert recorded["workspace_uri"] == "sqlite:///explicit.db"

    recorded.clear()
    monkeypatch.delenv(MLFLOW_WORKSPACE_URI.name, raising=False)
    client = MlflowClient()
    client.list_workspaces()
    # Falls back to the tracking URI when workspace URI is unset.
    assert recorded["workspace_uri"] == "sqlite:///tracking.db"
