import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI


@pytest.fixture(autouse=True)
def use_sqlite_if_tracking_uri_env_var_is_not_set(tmp_path, monkeypatch):
    if not MLFLOW_TRACKING_URI.defined:
        sqlite_file = tmp_path / "mlruns.sqlite"
        monkeypatch.setenv(MLFLOW_TRACKING_URI.name, f"sqlite:///{sqlite_file}")
