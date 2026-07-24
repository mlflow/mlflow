import pytest


@pytest.fixture(autouse=True)
def _clear_remote_env(monkeypatch):
    # A leaked MLFLOW_ALLOW_REMOTE_ASSISTANT would silently invert the
    # full_access tests (remote mode force-disables full_access). Tests that
    # exercise remote mode set it explicitly.
    monkeypatch.delenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", raising=False)
