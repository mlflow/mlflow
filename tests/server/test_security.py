import pytest
from flask import Flask
from werkzeug.test import Client

from mlflow.server import security
from mlflow.server.security_utils import is_allowed_host_header


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


def test_custom_allowed_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "example.com,app.example.com")
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
def test_dns_rebinding_protection(
    test_app, host_header, expected_status, expected_error, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost,127.0.0.1")
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
def test_cors_protection(
    test_app, method, origin, expected_cors_header, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "http://localhost:3000,https://app.example.com"
    )
    security.init_security_middleware(test_app)
    client = Client(test_app)

    headers = {"Origin": origin} if origin else {}
    response = getattr(client, method.lower())("/api/test", headers=headers)
    assert response.status_code == 200

    if expected_cors_header:
        assert response.headers.get("Access-Control-Allow-Origin") == expected_cors_header


def test_insecure_cors_mode(test_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "*")
    security.init_security_middleware(test_app)
    client = Client(test_app)

    response = client.post("/api/test", headers={"Origin": "http://evil.com"})
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "http://evil.com"


@pytest.mark.parametrize(
    ("origin", "expected_cors_header"),
    [
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://evil.com", None),
    ],
)
def test_preflight_options_request(
    test_app, origin, expected_cors_header, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    security.init_security_middleware(test_app)
    client = Client(test_app)

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


def test_disable_security_middleware(test_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    security.init_security_middleware(test_app)
    client = Client(test_app)

    response = client.get("/test")
    assert "X-Content-Type-Options" not in response.headers
    assert "X-Frame-Options" not in response.headers

    response = client.get("/test", headers={"Host": "evil.com"})
    assert response.status_code == 200


def test_x_frame_options_configuration(monkeypatch: pytest.MonkeyPatch):
    app = Flask(__name__)

    @app.route("/test")
    def test():
        return "OK"

    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "DENY")
    security.init_security_middleware(app)
    client = Client(app)
    response = client.get("/test")
    assert response.headers.get("X-Frame-Options") == "DENY"

    app2 = Flask(__name__)

    @app2.route("/test")
    def test2():
        return "OK"

    # Reset for the second app
    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "NONE")
    security.init_security_middleware(app2)
    client = Client(app2)
    response = client.get("/test")
    assert "X-Frame-Options" not in response.headers


def test_notebook_trace_renderer_skips_x_frame_options(monkeypatch: pytest.MonkeyPatch):
    from mlflow.tracing.constant import TRACE_RENDERER_ASSET_PATH

    app = Flask(__name__)

    @app.route(f"{TRACE_RENDERER_ASSET_PATH}/index.html")
    def notebook_renderer():
        return "<html>trace renderer</html>"

    @app.route(f"{TRACE_RENDERER_ASSET_PATH}/js/main.js")
    def notebook_renderer_js():
        return "console.log('trace renderer');"

    @app.route("/static-files/other-page.html")
    def other_page():
        return "<html>other page</html>"

    # Set X-Frame-Options to DENY to test that it's skipped for notebook renderer
    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "DENY")
    security.init_security_middleware(app)
    client = Client(app)

    response = client.get(f"{TRACE_RENDERER_ASSET_PATH}/index.html")
    assert response.status_code == 200
    assert "X-Frame-Options" not in response.headers

    response = client.get(f"{TRACE_RENDERER_ASSET_PATH}/js/main.js")
    assert response.status_code == 200
    assert "X-Frame-Options" not in response.headers

    response = client.get("/static-files/other-page.html")
    assert response.status_code == 200
    assert response.headers.get("X-Frame-Options") == "DENY"


def test_wildcard_hosts(test_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "*")
    security.init_security_middleware(test_app)
    client = Client(test_app)

    response = client.get("/test", headers={"Host": "any.domain.com"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("endpoint", "host_header", "expected_status"),
    [
        ("/health", "evil.com", 200),
        ("/test", "evil.com", 403),
    ],
)
def test_endpoint_security_bypass(
    test_app, endpoint, host_header, expected_status, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost")
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
    assert is_allowed_host_header(hosts, hostname) == expected_valid


@pytest.mark.parametrize(
    ("env_var", "env_value", "expected_result"),
    [
        (
            "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS",
            "http://app1.com,http://app2.com",
            ["http://app1.com", "http://app2.com"],
        ),
        ("MLFLOW_SERVER_ALLOWED_HOSTS", "app1.com,app2.com:8080", ["app1.com", "app2.com:8080"]),
    ],
)
def test_environment_variable_configuration(
    env_var, env_value, expected_result, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env_var, env_value)
    if "ORIGINS" in env_var:
        result = security.get_allowed_origins()
        for expected in expected_result:
            assert expected in result
    else:
        result = security.get_allowed_hosts()
        for expected in expected_result:
            assert expected in result
