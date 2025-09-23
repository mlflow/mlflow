import os
from unittest import mock

import pytest
from werkzeug.test import Client

from mlflow.server.security import SecurityMiddleware, init_security_middleware


def test_default_allowed_hosts():
    middleware = SecurityMiddleware()
    assert "localhost" in middleware.allowed_hosts
    assert "127.0.0.1" in middleware.allowed_hosts
    assert "[::1]" in middleware.allowed_hosts
    assert "localhost:5000" in middleware.allowed_hosts
    assert "127.0.0.1:5000" in middleware.allowed_hosts


def test_custom_allowed_hosts():
    middleware = SecurityMiddleware(allowed_hosts=["example.com", "app.example.com"])
    assert "example.com" in middleware.allowed_hosts
    assert "app.example.com" in middleware.allowed_hosts
    assert "localhost" not in middleware.allowed_hosts


@pytest.mark.parametrize(
    ("host_header", "expected_status", "expected_error"),
    [
        ("localhost", 200, None),
        ("127.0.0.1", 200, None),
        ("evil.attacker.com", 403, b"Invalid Host header"),
    ],
)
def test_dns_rebinding_protection(
    test_app, setup_middleware, host_header, expected_status, expected_error
):
    middleware = SecurityMiddleware(
        allowed_hosts=["localhost", "127.0.0.1"],
        enable_host_validation=True,
    )
    setup_middleware(middleware)
    client = Client(test_app)

    response = client.get("/test", headers={"Host": host_header})
    assert response.status_code == expected_status
    if expected_error:
        assert expected_error in response.data


@pytest.mark.parametrize(
    ("method", "origin", "expected_status", "expected_cors_header"),
    [
        ("POST", "http://localhost:3000", 200, "http://localhost:3000"),
        ("POST", "http://evil.com", 403, None),
        ("POST", None, 200, None),
        ("GET", "http://evil.com", 200, None),
    ],
)
def test_cors_protection(
    test_app, setup_middleware, method, origin, expected_status, expected_cors_header
):
    middleware = SecurityMiddleware(
        allowed_origins=["http://localhost:3000", "https://app.example.com"]
    )
    setup_middleware(middleware)
    client = Client(test_app)

    headers = {"Origin": origin} if origin else {}
    response = getattr(client, method.lower())("/api/test", headers=headers)
    assert response.status_code == expected_status

    if expected_cors_header:
        assert response.headers.get("Access-Control-Allow-Origin") == expected_cors_header

    if expected_status == 403:
        assert b"Cross-origin request blocked" in response.data


def test_insecure_cors_mode(test_app, setup_middleware):
    middleware = SecurityMiddleware(allow_insecure_cors=True)
    setup_middleware(middleware)
    client = Client(test_app)

    response = client.post("/api/test", headers={"Origin": "http://evil.com"})
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "*"


@pytest.mark.parametrize(
    ("origin", "expected_status", "expected_cors_header"),
    [
        ("http://localhost:3000", 204, "http://localhost:3000"),
        ("http://evil.com", 200, None),
    ],
)
def test_preflight_options_request(
    test_app, setup_middleware, origin, expected_status, expected_cors_header
):
    middleware = SecurityMiddleware(allowed_origins=["http://localhost:3000"])
    setup_middleware(middleware)
    client = Client(test_app)

    response = client.options("/api/test", headers={"Origin": origin})
    assert response.status_code == expected_status

    if expected_cors_header:
        assert response.headers.get("Access-Control-Allow-Origin") == expected_cors_header
        assert "Access-Control-Allow-Methods" in response.headers


def test_security_headers(test_app, setup_middleware):
    middleware = SecurityMiddleware()
    setup_middleware(middleware)
    client = Client(test_app)

    response = client.get("/test")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"


@pytest.mark.parametrize(
    ("endpoint", "host_header", "expected_status"),
    [
        ("/health", "evil.com", 200),
        ("/test", "evil.com", 403),
    ],
)
def test_endpoint_security_bypass(
    test_app, setup_middleware, endpoint, host_header, expected_status
):
    middleware = SecurityMiddleware(allowed_hosts=["localhost"])
    setup_middleware(middleware)
    client = Client(test_app)

    response = client.get(endpoint, headers={"Host": host_header})
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("hostname", "is_private"),
    [
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("172.16.0.1", True),
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("192.168.1.1:8080", True),
        ("[::1]:8080", True),
    ],
)
def test_private_ip_detection(hostname, is_private):
    middleware = SecurityMiddleware()
    assert middleware._is_private_ip(hostname) == is_private


@pytest.mark.parametrize(
    ("env_vars", "expected_attrs"),
    [
        (
            {"MLFLOW_CORS_ALLOWED_ORIGINS": "http://app1.com,http://app2.com"},
            {"allowed_origins": {"http://app1.com", "http://app2.com"}},
        ),
        (
            {"MLFLOW_ALLOW_INSECURE_CORS": "true"},
            {"allow_insecure_cors": True},
        ),
        (
            {"MLFLOW_HOST_HEADER_VALIDATION": "false"},
            {"enable_host_validation": False},
        ),
        (
            {"MLFLOW_ALLOWED_HOSTS": "app1.com,app2.com:8080"},
            {"allowed_hosts": {"app1.com", "app2.com:8080"}},
        ),
    ],
)
def test_environment_variable_configuration(test_app, env_vars, expected_attrs):
    with mock.patch.dict(os.environ, env_vars):
        middleware = init_security_middleware(test_app)

        for attr, expected_value in expected_attrs.items():
            actual_value = getattr(middleware, attr)
            if isinstance(expected_value, set):
                assert expected_value.issubset(actual_value)
            else:
                assert actual_value == expected_value
