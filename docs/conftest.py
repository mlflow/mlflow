import pytest

import mlflow


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmp_path, monkeypatch):
    tracking_uri = "sqlite:///{}".format(tmp_path / "mlruns.sqlite")
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture(autouse=True)
def reset_active_experiment_id():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
