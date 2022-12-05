import pytest
import mlflow


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmp_path, monkeypatch):
    tracking_uri = "sqlite:///{}".format(tmp_path / "mlruns.sqlite")
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
