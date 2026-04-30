"""Tests for the MLflow GenAI pytest plugin."""

import pytest

import mlflow


@pytest.fixture
def _isolated_tracking(tmp_path):
    """Set up an isolated tracking URI for testing."""
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.set_tracking_uri(None)


def test_nested_run_has_parent_id(_isolated_tracking):
    """Test that nested runs get a parent run ID."""
    mlflow.set_experiment("pytest")
    with mlflow.start_run(run_name="parent") as parent_run:
        parent_id = parent_run.info.run_id
        with mlflow.start_run(run_name="child_test", nested=True) as child_run:
            assert child_run.info.run_id != parent_id
            client = mlflow.MlflowClient()
            child_info = client.get_run(child_run.info.run_id)
            assert child_info.data.tags.get("mlflow.parentRunId") == parent_id


def test_plugin_module_exports():
    """Test that the plugin module exports expected fixtures."""
    from mlflow.genai import pytest_plugin

    assert hasattr(pytest_plugin, "mlflow_run")
    assert hasattr(pytest_plugin, "mlflow_experiment_name")
    assert hasattr(pytest_plugin, "pytest_addoption")


def test_plugin_has_pytest11_entry_point():
    """Verify the plugin is registered as a pytest11 entry point."""
    from importlib.metadata import entry_points

    eps = entry_points(group="pytest11")
    mlflow_eps = [ep for ep in eps if ep.name == "mlflow-genai"]
    assert len(mlflow_eps) == 1
    assert mlflow_eps[0].value == "mlflow.genai.pytest_plugin"
