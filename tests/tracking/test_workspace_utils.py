import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE_STORE_URI
from mlflow.utils.workspace_utils import resolve_workspace_store_uri, set_workspace_store_uri


@pytest.fixture(autouse=True)
def _reset_workspace_uri(monkeypatch):
    set_workspace_store_uri(None)
    monkeypatch.delenv(MLFLOW_WORKSPACE_STORE_URI.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_URI.name, raising=False)
    yield
    set_workspace_store_uri(None)
    monkeypatch.delenv(MLFLOW_WORKSPACE_STORE_URI.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_URI.name, raising=False)


def test_resolve_workspace_uri_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv(MLFLOW_WORKSPACE_STORE_URI.name, "sqlite:///env-workspaces.db")
    result = resolve_workspace_store_uri("sqlite:///explicit.db")
    assert result == "sqlite:///explicit.db"


def test_resolve_workspace_uri_uses_configured_value(monkeypatch):
    set_workspace_store_uri("sqlite:///configured.db")
    result = resolve_workspace_store_uri(tracking_uri="sqlite:///tracking.db")
    assert result == "sqlite:///configured.db"


def test_resolve_workspace_uri_uses_environment(monkeypatch):
    monkeypatch.setenv(MLFLOW_WORKSPACE_STORE_URI.name, "sqlite:///env.db")
    result = resolve_workspace_store_uri(tracking_uri="file:///mlruns")
    assert result == "sqlite:///env.db"


def test_resolve_workspace_uri_defaults_to_tracking(monkeypatch):
    result = resolve_workspace_store_uri(tracking_uri="sqlite:///tracking-default.db")
    assert result == "sqlite:///tracking-default.db"
