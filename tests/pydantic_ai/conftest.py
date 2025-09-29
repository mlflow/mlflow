import pytest

import mlflow

from tests.tracing.helper import purge_traces


@pytest.fixture(autouse=True)
def reset_mlflow_autolog_and_traces():
    yield
    mlflow.pydantic_ai.autolog(disable=True)
    purge_traces()


@pytest.fixture(autouse=True)
def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture(autouse=True)
def mock_creds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-secret-key")
    monkeypatch.setenv("GEMINI_API_KEY", "my-secret-key")
