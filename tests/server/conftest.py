import contextlib
import json as _json_mod

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server.fastapi_security import init_fastapi_security
from mlflow.server.request_context import RequestShim, _Args, _current_request


@contextlib.contextmanager
def mock_request_context(
    path="/",
    method="GET",
    data=None,
    content_type=None,
    headers=None,
    view_args=None,
    json=None,
    query_string=None,
):
    json_data = None
    raw_data = b""
    if json is not None:
        json_data = json
        raw_data = _json_mod.dumps(json).encode("utf-8")
        if content_type is None:
            content_type = "application/json"
    elif data is not None:
        raw_data = data.encode("utf-8") if isinstance(data, str) else data
        if content_type and "json" in content_type:
            try:
                json_data = _json_mod.loads(raw_data)
            except (ValueError, TypeError):
                pass
    args_data: dict[str, list[str]] = {}
    if query_string is not None:
        if isinstance(query_string, dict):
            for k, v in query_string.items():
                args_data[k] = [str(v)]
        elif isinstance(query_string, list):
            for k, v in query_string:
                args_data.setdefault(k, []).append(str(v))
    shim = RequestShim(
        method=method,
        path=path,
        content_type=content_type,
        args=_Args(_data=args_data),
        _json=json_data,
        _data=raw_data,
        _headers=dict(headers or {}),
        view_args=dict(view_args or {}),
    )
    token = _current_request.set(shim)
    try:
        yield shim
    finally:
        _current_request.reset(token)


@pytest.fixture
def fastapi_client():
    """Minimal FastAPI app for unit testing."""
    app = FastAPI()

    @app.api_route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    async def api_endpoint():
        return {"ok": True}

    init_fastapi_security(app)
    return TestClient(app, raise_server_exceptions=False)
