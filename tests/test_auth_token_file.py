# tests/test_auth_token_file.py
import os
import tempfile
from unittest.mock import patch
import mlflow.utils.auth as auth
from mlflow.utils import rest_utils

def test_token_file_injected(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("test-token-123\n")
    monkeypatch.setenv("MLFLOW_AUTH_TOKEN_FILE", str(token_file))

    # Capture headers used in requests; monkeypatch requests.request
    import requests
    called = {}
    def fake_request(method, url, headers=None, **kwargs):
        called['headers'] = headers or {}
        class R:
            status_code = 200
            raw = type("R", (), {"headers": {}, "stream": None})
            def iter_content(self, chunk_size=1024): return []
        return R()
    monkeypatch.setattr(requests, "request", fake_request)

    # Call the code path that triggers http_request (adjust as needed to hit function)
    # Example: call rest_utils.http_request(...) with minimal args (depends on MLflow version)
    try:
        rest_utils.http_request("GET", "http://example", None)  # adapt to real signature
    except Exception:
        # the test focuses on headers; ignore other failures
        pass

    assert "Authorization" in called['headers']
    assert called['headers']["Authorization"] == "Bearer test-token-123"
