import pytest
import mlflow


@pytest.fixture(autouse=True)
def set_tracking_uri(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
