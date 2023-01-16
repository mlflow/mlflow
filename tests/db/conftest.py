import os

import pytest

from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR


@pytest.fixture(autouse=True)
def use_sqlite_if_tracking_uri_env_var_is_not_set(tmp_path, monkeypatch):
    if _TRACKING_URI_ENV_VAR not in os.environ:
        sqlite_file = tmp_path / "mlruns.sqlite"
        monkeypatch.setenv(_TRACKING_URI_ENV_VAR, f"sqlite:///{sqlite_file}")
