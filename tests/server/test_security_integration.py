import json

import pytest
from werkzeug.test import Client


@pytest.mark.parametrize(
    ("host", "origin", "expected_status", "should_block"),
    [
        ("evil.attacker.com:5000", "http://evil.attacker.com:5000", 403, True),
        ("localhost:5000", None, None, False),
    ],
)
def test_dns_rebinding_and_cors_protection(
    mlflow_app_client, host, origin, expected_status, should_block
):
    headers = {"Host": host, "Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin

    response = mlflow_app_client.post(
        "/api/2.0/mlflow/experiments/search",
        headers=headers,
        data=json.dumps({"order_by": ["creation_time DESC", "name ASC"], "max_results": 50}),
    )

    if should_block:
        assert response.status_code == expected_status
        assert (
            b"Invalid Host header" in response.data
            or b"Cross-origin request blocked" in response.data
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
def test_cors_for_state_changing_requests(mlflow_app_client, origin, endpoint, expected_blocked):
    response = mlflow_app_client.post(
        endpoint,
        headers={"Origin": origin, "Content-Type": "application/json"},
        data=json.dumps({"name": "test-experiment"} if "create" in endpoint else {}),
    )

    if expected_blocked:
        assert response.status_code == 403
        assert b"Cross-origin request blocked" in response.data
    else:
        assert response.status_code != 403


def test_cors_with_configured_origins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "https://trusted-app.com")

    from flask import Flask

    from mlflow.server import handlers, security

    app = Flask(__name__)
    for http_path, handler, methods in handlers.get_endpoints():
        app.add_url_rule(http_path, handler.__name__, handler, methods=methods)

    security.init_security_middleware(app)
    client = Client(app)

    test_cases = [
        ("https://trusted-app.com", False),
        ("http://evil.com", True),
    ]

    for origin, should_block in test_cases:
        response = client.post(
            "/api/2.0/mlflow/experiments/search",
            headers={"Origin": origin, "Content-Type": "application/json"},
            data=json.dumps({}),
        )

        if should_block:
            assert response.status_code == 403
        else:
            assert response.status_code != 403


def test_security_headers_on_responses(mlflow_app_client):
    response = mlflow_app_client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"


@pytest.mark.parametrize(
    ("origin", "expected_status", "should_have_cors"),
    [
        ("http://localhost:3000", 204, True),
        ("http://evil.com", None, False),
    ],
)
def test_preflight_options_requests(mlflow_app_client, origin, expected_status, should_have_cors):
    response = mlflow_app_client.options(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    if expected_status:
        assert response.status_code == expected_status

    if should_have_cors:
        assert response.headers.get("Access-Control-Allow-Origin") == origin
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")
    else:
        assert (
            "Access-Control-Allow-Origin" not in response.headers
            or response.headers.get("Access-Control-Allow-Origin") != origin
        )
