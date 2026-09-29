import threading

import anyio
import pytest
from flask import Flask
from starlette.testclient import TestClient

try:
    import httpx2 as httpx
except ModuleNotFoundError:
    import httpx

from mlflow.server import app as flask_app
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

_SERVER_INFO_PATHS = (
    "/api/3.0/mlflow/server-info",
    "/ajax-api/3.0/mlflow/server-info",
)


def _flask_server_info(path: str) -> dict:
    response = flask_app.test_client().get(path)
    assert response.status_code == 200
    return response.get_json()


def test_server_info_matches_flask_json(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    with TestClient(create_fastapi_app()) as client:
        for path in _SERVER_INFO_PATHS:
            response = client.get(path)
            assert response.status_code == 200
            assert response.json() == _flask_server_info(path)


def test_server_info_matches_flask_json_with_static_prefix(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")
    with TestClient(create_fastapi_app()) as client:
        for path in _SERVER_INFO_PATHS:
            response = client.get(f"/mlflow{path}")
            assert response.status_code == 200
            assert response.json() == _flask_server_info(path)


@pytest.mark.asyncio
async def test_server_info_responds_when_wsgi_pool_is_exhausted(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    started = threading.Event()
    release = threading.Event()
    blocking_app = Flask("wsgi-pool-block")

    @blocking_app.route("/_block")
    def _block():
        started.set()
        release.wait(timeout=30)
        return "ok"

    fastapi_app = create_fastapi_app(blocking_app)
    # The default limiter lives on the app's event loop. Shrink it there, and
    # issue the blocking Flask call and server-info as concurrent ASGI tasks.
    saved_limiter: list[tuple[anyio.CapacityLimiter, float]] = []

    @fastapi_app.middleware("http")
    async def _exhaust_default_limiter(request, call_next):
        limiter = anyio.to_thread.current_default_thread_limiter()
        if not saved_limiter:
            saved_limiter.append((limiter, limiter.total_tokens))
            limiter.total_tokens = 1
        return await call_next(request)

    response = None
    try:
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def _hold_wsgi_pool():
                await client.get("/_block")

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_hold_wsgi_pool)
                try:
                    for _ in range(100):
                        if started.is_set():
                            break
                        await anyio.sleep(0.05)
                    assert started.is_set()
                    with anyio.move_on_after(2):
                        response = await client.get("/api/3.0/mlflow/server-info")
                finally:
                    release.set()
    finally:
        if saved_limiter:
            limiter, original_tokens = saved_limiter[0]
            limiter.total_tokens = original_tokens

    assert response is not None, "server-info blocked on the shared WSGI thread pool"
    assert response.status_code == 200


def test_server_info_stays_public_when_fail_closed(monkeypatch):
    pytest.importorskip("flask_wtf")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    from mlflow.server.auth import add_fastapi_permission_middleware

    app = create_fastapi_app()
    add_fastapi_permission_middleware(app)
    with TestClient(app) as client:
        response = client.get("/api/3.0/mlflow/server-info")
        assert response.status_code == 200
        assert "authorization" not in {name.lower() for name in response.request.headers}
