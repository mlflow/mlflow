import json

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server import handlers
from mlflow.server.fastapi_app import _flask_to_fastapi_path, add_request_shim_middleware
from mlflow.server.fastapi_security import init_fastapi_security
from mlflow.server.responses import Response


def _make_app():
    app = FastAPI()
    add_request_shim_middleware(app)
    for http_path, handler, methods in handlers.get_endpoints():
        fastapi_path = _flask_to_fastapi_path(http_path)
        app.add_api_route(fastapi_path, handler, methods=methods, response_class=Response)
    init_fastapi_security(app)
    return app


@pytest.fixture
def mlflow_fastapi_client():
    return TestClient(_make_app(), raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("host", "origin", "expected_status", "should_block"),
    [
        ("evil.attacker.com:5000", "http://evil.attacker.com:5000", 403, True),
        ("localhost:5000", None, None, False),
    ],
)
def test_dns_rebinding_and_cors_protection(
    mlflow_fastapi_client, host, origin, expected_status, should_block
):
    headers = {"Host": host, "Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin

    response = mlflow_fastapi_client.post(
        "/api/2.0/mlflow/experiments/search",
        headers=headers,
        content=json.dumps({"order_by": ["creation_time DESC", "name ASC"], "max_results": 50}),
    )

    if should_block:
        assert response.status_code == expected_status
        assert (
            b"Invalid Host header" in response.content
            or b"Cross-origin request blocked" in response.content
        )
    else:
        assert response.status_code != 403


@pytest.mark.parametrize(
    ("origin", "endpoint", "expected_blocked"),
    [
        ("http://malicious-site.com", "/api/2.0/mlflow/experiments/create", True),
        ("http://localhost:3000", "/api/2.0/mlflow/experiments/search", False),
    ],
)
def test_cors_for_state_changing_requests(
    mlflow_fastapi_client, origin, endpoint, expected_blocked
):
    response = mlflow_fastapi_client.post(
        endpoint,
        headers={"Host": "localhost", "Origin": origin, "Content-Type": "application/json"},
        content=json.dumps({"name": "test-experiment"} if "create" in endpoint else {}),
    )

    if expected_blocked:
        assert response.status_code == 403
        assert b"Cross-origin request blocked" in response.content
    else:
        assert response.status_code != 403


def test_cors_with_configured_origins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "https://trusted-app.com")

    client = TestClient(_make_app(), raise_server_exceptions=False)

    test_cases = [
        ("https://trusted-app.com", False),
        ("http://evil.com", True),
    ]

    for origin, should_block in test_cases:
        response = client.post(
            "/api/2.0/mlflow/experiments/search",
            headers={
                "Host": "localhost",
                "Origin": origin,
                "Content-Type": "application/json",
            },
            content=json.dumps({}),
        )

        if should_block:
            assert response.status_code == 403
        else:
            assert response.status_code != 403


def test_security_headers_on_responses(mlflow_fastapi_client):
    response = mlflow_fastapi_client.get("/health", headers={"Host": "localhost"})
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"


@pytest.mark.parametrize(
    ("origin", "expected_status", "should_have_cors"),
    [
        ("http://localhost:3000", 200, True),
        ("http://evil.com", None, False),
    ],
)
def test_preflight_options_requests(
    mlflow_fastapi_client, origin, expected_status, should_have_cors
):
    response = mlflow_fastapi_client.options(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Host": "localhost",
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    if expected_status:
        assert response.status_code == expected_status

    if should_have_cors:
        assert response.headers.get("access-control-allow-origin") == origin
        assert "POST" in response.headers.get("access-control-allow-methods", "")
    else:
        assert (
            "access-control-allow-origin" not in response.headers
            or response.headers.get("access-control-allow-origin") != origin
        )
