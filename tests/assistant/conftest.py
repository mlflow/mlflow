import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_REMOTE_ASSISTANT


@pytest.fixture(autouse=True)
def _clear_remote_env(monkeypatch):
    # A leaked MLFLOW_ENABLE_REMOTE_ASSISTANT would silently invert the
    # full_access tests (remote mode force-disables full_access). Tests that
    # exercise remote mode set it explicitly.
    monkeypatch.delenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, raising=False)
