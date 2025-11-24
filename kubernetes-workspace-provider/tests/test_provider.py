from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kubernetes_workspace_provider.provider import (
    KubernetesWorkspaceProvider,
    create_kubernetes_workspace_store,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR", raising=False)
    monkeypatch.delenv("MLFLOW_K8S_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS", raising=False)


@pytest.fixture
def core_api(monkeypatch):
    mock_core = MagicMock()
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.config.load_kube_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.config.load_incluster_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.client.CoreV1Api",
        lambda: mock_core,
    )

    class _FakeWatch:
        def stream(self, *args, **kwargs):
            return iter(())

        def stop(self):
            return None

    monkeypatch.setattr(
        "kubernetes_workspace_provider.provider.watch.Watch",
        lambda: _FakeWatch(),
    )
    return mock_core


def _namespace(name: str, description: str | None = None):
    annotations = {"mlflow.kubeflow.org/workspace-description": description} if description else {}
    metadata = SimpleNamespace(name=name, annotations=annotations, resource_version="1")
    return SimpleNamespace(metadata=metadata)


def test_list_workspaces_uses_cache(core_api):
    namespaces = [_namespace("team-a", "Team A"), _namespace("team-b")]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="5"),
    )

    provider = KubernetesWorkspaceProvider()

    first = provider.list_workspaces()
    second = provider.list_workspaces()

    assert core_api.list_namespace.call_args_list[0][1]["label_selector"] is None
    assert [ws.name for ws in first] == ["team-a", "team-b"]
    assert [ws.description for ws in first] == ["Team A", None]
    assert [ws.name for ws in second] == ["team-a", "team-b"]


def test_system_namespaces_are_filtered(core_api):
    namespaces = [
        _namespace("kube-system"),
        _namespace("openshift-config"),
        _namespace("team-a"),
    ]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="6"),
    )

    provider = KubernetesWorkspaceProvider()

    assert [ws.name for ws in provider.list_workspaces()] == ["team-a"]


def test_custom_namespace_filter_from_env(core_api, monkeypatch):
    monkeypatch.setenv("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS", "secret-*,*-internal")
    namespaces = [
        _namespace("team-a"),
        _namespace("secret-workspace"),
        _namespace("ml-internal"),
    ]
    core_api.list_namespace.return_value = SimpleNamespace(
        items=namespaces,
        metadata=SimpleNamespace(resource_version="7"),
    )

    provider = KubernetesWorkspaceProvider()

    assert [ws.name for ws in provider.list_workspaces()] == ["team-a"]


def test_get_workspace_reads_namespace(core_api):
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("analytics", "Analytics Workspace")],
        metadata=SimpleNamespace(resource_version="9"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_workspace("analytics")

    assert not core_api.read_namespace.called
    assert workspace.name == "analytics"
    assert workspace.description == "Analytics Workspace"


def test_get_default_workspace_env(core_api, monkeypatch):
    monkeypatch.setenv("MLFLOW_K8S_DEFAULT_WORKSPACE", "shared")
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("shared", "Shared")],
        metadata=SimpleNamespace(resource_version="17"),
    )

    provider = KubernetesWorkspaceProvider()
    workspace = provider.get_default_workspace()

    assert workspace.name == "shared"


def test_get_default_workspace_requires_selection(core_api):
    core_api.list_namespace.return_value = SimpleNamespace(
        items=[_namespace("alpha")],
        metadata=SimpleNamespace(resource_version="21"),
    )

    provider = KubernetesWorkspaceProvider()

    with pytest.raises(NotImplementedError, match="Active workspace is required"):
        provider.get_default_workspace()


def test_create_workspace_store_parses_uri_options(core_api):
    store = create_kubernetes_workspace_store(
        "kubernetes://?label_selector=team%3Dmlflow&default_workspace=shared"
        "&namespace_exclude_globs=team-secret-*,%20extra"
    )

    assert isinstance(store, KubernetesWorkspaceProvider)
    assert store._config.label_selector == "team=mlflow"
    assert store._config.default_workspace == "shared"
    assert store._config.namespace_exclude_globs == (
        "kube-*",
        "openshift-*",
        "team-secret-*",
        "extra",
    )
