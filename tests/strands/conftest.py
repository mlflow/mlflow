import os

import pytest

import mlflow

from tests.helper_functions import start_mock_openai_server


@pytest.fixture(autouse=True)
def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in list(AUTOLOGGING_INTEGRATIONS.keys()):
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}
    import opentelemetry.trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    trace_api.set_tracer_provider(TracerProvider())


@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("PYTHONPATH", os.getcwd())
    with start_mock_openai_server() as base_url:
        yield base_url
