from __future__ import annotations

from mlflow import MlflowClient
from mlflow.entities import TraceArchivalConfig
from mlflow.environment_variables import MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE_STORE_URI
from mlflow.tracking._workspace import fluent as workspace_fluent
from mlflow.tracking._workspace.client import WorkspaceProviderClient
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


def test_mlflow_client_forwards_trace_archival_settings(monkeypatch):
    recorded: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    class DummyTrackingClient:
        def __init__(self, tracking_uri: str):
            self.tracking_uri = tracking_uri

    class DummyWorkspaceClient:
        def __init__(self, workspace_uri: str | None = None):
            self.workspace_uri = workspace_uri

        def create_workspace(self, *args, **kwargs):
            recorded["create"] = (args, kwargs)
            return None

        def update_workspace(self, *args, **kwargs):
            recorded["update"] = (args, kwargs)
            return None

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

    client = MlflowClient(tracking_uri="sqlite:///tracking.db")
    client.create_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(
            location="s3://archive/team-a",
            retention="30d",
        ),
    )
    client.update_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(
            location="s3://archive/team-b",
            retention="14d",
        ),
    )

    assert recorded["create"][1]["trace_archival_config"] == TraceArchivalConfig(
        location="s3://archive/team-a",
        retention="30d",
    )
    assert recorded["update"][1]["trace_archival_config"] == TraceArchivalConfig(
        location="s3://archive/team-b",
        retention="14d",
    )


def test_workspace_fluent_forwards_trace_archival_config(monkeypatch):
    recorded: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    class DummyClient:
        def create_workspace(self, *args, **kwargs):
            recorded["create"] = (args, kwargs)
            return None

        def update_workspace(self, *args, **kwargs):
            recorded["update"] = (args, kwargs)
            return None

    monkeypatch.setattr(workspace_fluent, "MlflowClient", DummyClient)

    workspace_fluent.create_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(
            location="s3://archive/team-a",
            retention="30d",
        ),
    )
    workspace_fluent.update_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(
            location="",
            retention="",
        ),
    )

    assert recorded["create"][1]["trace_archival_config"] == TraceArchivalConfig(
        location="s3://archive/team-a",
        retention="30d",
    )
    assert recorded["update"][1]["trace_archival_config"] == TraceArchivalConfig(
        location="",
        retention="",
    )


def test_workspace_provider_client_flattens_trace_archival_config(monkeypatch):
    recorded: dict[str, object] = {}

    class DummyStore:
        def create_workspace(self, workspace):
            recorded["create"] = workspace
            return workspace

        def update_workspace(self, workspace):
            recorded["update"] = workspace
            return workspace

    monkeypatch.setattr(
        "mlflow.tracking._workspace.client.get_workspace_store",
        lambda workspace_uri: DummyStore(),
    )

    client = WorkspaceProviderClient("sqlite:///workspace.db")
    created = client.create_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(location="s3://archive/team-a"),
    )
    updated = client.update_workspace(
        "team-a",
        trace_archival_config=TraceArchivalConfig(location="", retention=""),
    )

    assert recorded["create"] == created
    assert created.trace_archival_location == "s3://archive/team-a"
    assert created.trace_archival_retention is None

    assert recorded["update"] == updated
    assert updated.trace_archival_location == ""
    assert updated.trace_archival_retention == ""
