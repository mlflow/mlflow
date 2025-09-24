import os
from unittest import mock

import pytest
from werkzeug.test import Client

from mlflow.server import security


def test_default_allowed_hosts():
    hosts = security.get_allowed_hosts()
    assert "localhost" in hosts
    assert "127.0.0.1" in hosts
    assert "[::1]" in hosts
    assert "localhost:*" in hosts
    assert "127.0.0.1:*" in hosts
    assert "[[]::1]:*" in hosts
    assert "192.168.*" in hosts
    assert "10.*" in hosts


def test_custom_allowed_hosts():
    with mock.patch.dict(os.environ, {"MLFLOW_ALLOWED_HOSTS": "example.com,app.example.com"}):
        hosts = security.get_allowed_hosts()
        assert "example.com" in hosts
        assert "app.example.com" in hosts


@pytest.mark.parametrize(
    ("host_header", "expected_status", "expected_error"),
    [
        ("localhost", 200, None),
        ("127.0.0.1", 200, None),
        ("evil.attacker.com", 403, b"Invalid Host header"),
    ],
)
def test_dns_rebinding_protection(test_app, host_header, expected_status, expected_error):
    with mock.patch.dict(os.environ, {"MLFLOW_ALLOWED_HOSTS": "localhost,127.0.0.1"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)

        response = client.get("/test", headers={"Host": host_header})
        assert response.status_code == expected_status
        if expected_error:
            assert expected_error in response.data


@pytest.mark.parametrize(
    ("method", "origin", "expected_cors_header"),
    [
        ("POST", "http://localhost:3000", "http://localhost:3000"),
        ("POST", "http://evil.com", None),
        ("POST", None, None),
        ("GET", "http://evil.com", None),
    ],
)
def test_cors_protection(test_app, method, origin, expected_cors_header):
    with mock.patch.dict(
        os.environ, {"MLFLOW_CORS_ALLOWED_ORIGINS": "http://localhost:3000,https://app.example.com"}
    ):
        security.init_security_middleware(test_app)
        client = Client(test_app)

        headers = {"Origin": origin} if origin else {}
        response = getattr(client, method.lower())("/api/test", headers=headers)
        assert response.status_code == 200

        if expected_cors_header:
            assert response.headers.get("Access-Control-Allow-Origin") == expected_cors_header


def test_insecure_cors_mode(test_app):
    with mock.patch.dict(os.environ, {"MLFLOW_ALLOW_INSECURE_CORS": "true"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)

        response = client.post("/api/test", headers={"Origin": "http://evil.com"})
        assert response.status_code == 200
        # In insecure mode, Flask-CORS returns the requested origin
        assert response.headers.get("Access-Control-Allow-Origin") == "http://evil.com"


@pytest.mark.parametrize(
    ("origin", "expected_cors_header"),
    [
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://evil.com", None),
    ],
)
def test_preflight_options_request(test_app, origin, expected_cors_header):
    with mock.patch.dict(os.environ, {"MLFLOW_CORS_ALLOWED_ORIGINS": "http://localhost:3000"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)

        # For preflight, we need to send proper CORS preflight headers
        response = client.options(
            "/api/test",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200

        if expected_cors_header:
            assert response.headers.get("Access-Control-Allow-Origin") == expected_cors_header


def test_security_headers(test_app):
    security.init_security_middleware(test_app)
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
def test_endpoint_security_bypass(test_app, endpoint, host_header, expected_status):
    with mock.patch.dict(os.environ, {"MLFLOW_ALLOWED_HOSTS": "localhost"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)

        response = client.get(endpoint, headers={"Host": host_header})
        assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("hostname", "expected_valid"),
    [
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("172.16.0.1", True),
        ("127.0.0.1", True),
        ("localhost", True),
        ("[::1]", True),
        ("192.168.1.1:8080", True),
        ("[::1]:8080", True),
        ("evil.com", False),
    ],
)
def test_host_validation(hostname, expected_valid):
    hosts = security.get_allowed_hosts()
    assert security.validate_host_header(hosts, hostname) == expected_valid


@pytest.mark.parametrize(
    ("env_var", "env_value", "expected_result"),
    [
        (
            "MLFLOW_CORS_ALLOWED_ORIGINS",
            "http://app1.com,http://app2.com",
            ["http://app1.com", "http://app2.com"],
        ),
        ("MLFLOW_ALLOWED_HOSTS", "app1.com,app2.com:8080", ["app1.com", "app2.com:8080"]),
    ],
)
def test_environment_variable_configuration(env_var, env_value, expected_result):
    with mock.patch.dict(os.environ, {env_var: env_value}):
        if "ORIGINS" in env_var:
            result = security.get_allowed_origins()
            for expected in expected_result:
                assert expected in result
        else:
            result = security.get_allowed_hosts()
            for expected in expected_result:
                assert expected in result


def test_insecure_cors_flag(test_app):
    with mock.patch.dict(os.environ, {"MLFLOW_ALLOW_INSECURE_CORS": "true"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)
        response = client.get("/test", headers={"Origin": "http://any.site.com"})
        # Flask-CORS returns the origin even in insecure mode, not '*'
        assert response.headers.get("Access-Control-Allow-Origin") == "http://any.site.com"


def test_host_validation_disabled(test_app):
    with mock.patch.dict(os.environ, {"MLFLOW_HOST_HEADER_VALIDATION": "false"}):
        security.init_security_middleware(test_app)
        client = Client(test_app)
        response = client.get("/test", headers={"Host": "evil.com"})
        assert response.status_code == 200
