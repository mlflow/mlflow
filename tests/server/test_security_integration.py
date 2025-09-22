"""
Integration tests for MLflow server security against DNS rebinding attacks.
"""

import json
import pytest
from unittest import mock

from mlflow.server import app
from werkzeug.test import Client


def test_dns_rebinding_protection_blocks_malicious_requests():
    """
    Test that the MLflow server blocks DNS rebinding attack attempts.

    This simulates the attack described in the security report where
    a malicious website attempts to access MLflow endpoints using
    DNS rebinding.
    """
    client = Client(app)

    # Simulate a malicious DNS rebinding attack
    # The attacker's website would try to make requests with an external host header
    malicious_host = "evil.attacker.com:5000"

    # Try to access the experiments search endpoint (as described in the attack)
    response = client.post(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Host": malicious_host,
            "Origin": f"http://{malicious_host}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"order_by": ["creation_time DESC", "name ASC"], "max_results": 50}),
    )

    # The request should be blocked
    assert response.status_code == 403
    assert (
        b"Invalid Host header" in response.data or b"Cross-origin request blocked" in response.data
    )

    # Verify that legitimate localhost requests still work
    response = client.post(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Host": "localhost:5000",
            "Content-Type": "application/json",
        },
        data=json.dumps({"order_by": ["creation_time DESC", "name ASC"], "max_results": 50}),
    )

    # This should work (might return 400 if no backend is configured, but not 403)
    assert response.status_code != 403


def test_cors_protection_blocks_unauthorized_origins():
    """
    Test that CORS protection blocks requests from unauthorized origins.
    """
    client = Client(app)

    # Try to make a state-changing request from an unauthorized origin
    response = client.post(
        "/api/2.0/mlflow/experiments/create",
        headers={
            "Origin": "http://malicious-site.com",
            "Content-Type": "application/json",
        },
        data=json.dumps({"name": "test-experiment"}),
    )

    # Should be blocked by CORS protection
    assert response.status_code == 403
    assert b"Cross-origin request blocked" in response.data


@mock.patch.dict("os.environ", {"MLFLOW_CORS_ALLOWED_ORIGINS": "https://trusted-app.com"})
def test_cors_with_allowed_origins():
    """
    Test that CORS works correctly with configured allowed origins.
    """
    # Need to reimport to pick up the new environment variable
    import importlib
    import mlflow.server

    importlib.reload(mlflow.server)

    client = Client(mlflow.server.app)

    # Request from allowed origin should work
    response = client.post(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Origin": "https://trusted-app.com",
            "Content-Type": "application/json",
        },
        data=json.dumps({}),
    )

    # Should not be blocked by CORS (might return 400 for bad request, but not 403)
    assert response.status_code != 403

    # Request from non-allowed origin should be blocked
    response = client.post(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Origin": "http://evil.com",
            "Content-Type": "application/json",
        },
        data=json.dumps({}),
    )

    assert response.status_code == 403


def test_security_headers_are_present():
    """
    Test that security headers are added to responses.
    """
    client = Client(app)

    response = client.get("/health")

    # Check for security headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"


def test_preflight_requests_handled_correctly():
    """
    Test that OPTIONS preflight requests are handled correctly.
    """
    client = Client(app)

    # Preflight from allowed origin (localhost)
    response = client.options(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 204
    assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

    # Preflight from disallowed origin
    response = client.options(
        "/api/2.0/mlflow/experiments/search",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    # Should not have CORS headers for disallowed origin
    assert (
        "Access-Control-Allow-Origin" not in response.headers
        or response.headers.get("Access-Control-Allow-Origin") != "http://evil.com"
    )
