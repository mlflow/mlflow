import pytest

import mlflow


@pytest.fixture(autouse=True)
def _reset_mlflow():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for integ in AUTOLOGGING_INTEGRATIONS.values():
        integ.clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture(autouse=True)
def mock_creds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
