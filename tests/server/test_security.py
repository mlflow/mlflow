import logging

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server.fastapi_security import (
    get_allowed_hosts,
    get_allowed_origins,
    init_fastapi_security,
)
from mlflow.server.security_utils import is_allowed_host_header, is_api_endpoint


def _make_test_app():
    app = FastAPI()

    @app.api_route("/test", methods=["GET", "POST"])
    async def test_endpoint():
        return "OK"

    @app.api_route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    async def api_endpoint():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return "OK"

    @app.get("/version")
    async def version():
        return "OK"

    return app


def test_default_allowed_hosts():
    hosts = get_allowed_hosts()
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
    hosts = get_allowed_hosts()
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
    host_header, expected_status, expected_error, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost,127.0.0.1")
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/test", headers={"Host": host_header})
    assert response.status_code == expected_status
    if expected_error:
        assert expected_error in response.content


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
    method, origin, expected_status, expected_cors_header, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "http://localhost:3000,https://app.example.com"
    )
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    headers = {"Host": "localhost"}
    if origin:
        headers["Origin"] = origin
    response = getattr(client, method.lower())("/api/2.0/mlflow/experiments/list", headers=headers)
    assert response.status_code == expected_status

    if expected_cors_header:
        assert response.headers.get("access-control-allow-origin") == expected_cors_header


def test_wildcard_cors_disables_credentials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "*")
    security_logger = logging.getLogger("mlflow.server.fastapi_security")
    security_logger.addHandler(caplog.handler)

    app = _make_test_app()

    try:
        with caplog.at_level("WARNING", logger="mlflow.server.fastapi_security"):
            init_fastapi_security(app)
    finally:
        security_logger.removeHandler(caplog.handler)
    assert any("disabling credentialed CORS" in record.message for record in caplog.records)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/2.0/mlflow/experiments/list",
        headers={"Host": "localhost", "Origin": "http://evil.com"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-credentials") != "true"

    preflight = client.options(
        "/api/2.0/mlflow/experiments/list",
        headers={
            "Host": "localhost",
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.headers.get("access-control-allow-credentials") != "true"


@pytest.mark.parametrize(
    ("origin", "expected_status", "expected_cors_header"),
    [
        ("http://localhost:3000", 200, "http://localhost:3000"),
        ("http://evil.com", 400, None),
    ],
)
def test_preflight_options_request(
    origin, expected_status, expected_cors_header, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.options(
        "/api/2.0/mlflow/experiments/list",
        headers={
            "Host": "localhost",
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == expected_status

    if expected_cors_header:
        assert response.headers.get("access-control-allow-origin") == expected_cors_header


def test_security_headers():
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/test", headers={"Host": "localhost"})
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"


def test_disable_security_middleware(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/test")
    assert "x-content-type-options" not in response.headers
    assert "x-frame-options" not in response.headers

    response = client.get("/test", headers={"Host": "evil.com"})
    assert response.status_code == 200


def test_x_frame_options_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "DENY")
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test", headers={"Host": "localhost"})
    assert response.headers.get("x-frame-options") == "DENY"

    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "NONE")
    app2 = _make_test_app()
    init_fastapi_security(app2)
    client2 = TestClient(app2, raise_server_exceptions=False)
    response = client2.get("/test", headers={"Host": "localhost"})
    assert "x-frame-options" not in response.headers


def test_notebook_trace_renderer_skips_x_frame_options(monkeypatch: pytest.MonkeyPatch):
    from mlflow.tracing.constant import TRACE_RENDERER_ASSET_PATH

    app = FastAPI()

    @app.get(f"{TRACE_RENDERER_ASSET_PATH}/index.html")
    async def notebook_renderer():
        return "<html>trace renderer</html>"

    @app.get(f"{TRACE_RENDERER_ASSET_PATH}/js/main.js")
    async def notebook_renderer_js():
        return "console.log('trace renderer');"

    @app.get("/static-files/other-page.html")
    async def other_page():
        return "<html>other page</html>"

    monkeypatch.setenv("MLFLOW_SERVER_X_FRAME_OPTIONS", "DENY")
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        f"{TRACE_RENDERER_ASSET_PATH}/index.html", headers={"Host": "localhost"}
    )
    assert response.status_code == 200
    assert "x-frame-options" not in response.headers

    response = client.get(
        f"{TRACE_RENDERER_ASSET_PATH}/js/main.js", headers={"Host": "localhost"}
    )
    assert response.status_code == 200
    assert "x-frame-options" not in response.headers

    response = client.get("/static-files/other-page.html", headers={"Host": "localhost"})
    assert response.status_code == 200
    assert response.headers.get("x-frame-options") == "DENY"


@pytest.mark.parametrize(
    ("allowed_hosts", "host_header", "expected_status"),
    [
        ("*", "any.domain.com", 200),
        ("*.example.com", "app.example.com", 200),
        ("*.example.com", "sub.app.example.com", 200),
        ("*.example.com", "evil.com", 403),
    ],
)
def test_wildcard_hosts(
    allowed_hosts, host_header, expected_status, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", allowed_hosts)
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/test", headers={"Host": host_header})
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("allowed_origins", "origin", "expected_status"),
    [
        ("*", "http://any.domain.com", 200),
        ("http://*.example.com", "http://app.example.com", 200),
        ("http://*.example.com", "http://sub.app.example.com", 200),
        ("http://*.example.com", "http://evil.com", 403),
    ],
)
def test_wildcard_origins(
    allowed_origins, origin, expected_status, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", allowed_origins)
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/2.0/mlflow/experiments/list",
        headers={"Host": "localhost", "Origin": origin},
    )
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("endpoint", "host_header", "expected_status"),
    [
        ("/health", "evil.com", 200),
        ("/test", "evil.com", 403),
    ],
)
def test_endpoint_security_bypass(
    endpoint, host_header, expected_status, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost")
    app = _make_test_app()
    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

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
    hosts = get_allowed_hosts()
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
        result = get_allowed_origins()
        for expected in expected_result:
            assert expected in result
    else:
        result = get_allowed_hosts()
        for expected in expected_result:
            assert expected in result


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/2.0/mlflow/experiments/list", True),
        ("/ajax-api/2.0/mlflow/experiments/list", True),
        ("/ajax-api/3.0/mlflow/runs/search", True),
        ("/api/test", False),
        ("/test", False),
        ("/health", False),
        ("/static/index.html", False),
    ],
)
def test_is_api_endpoint(path, expected):
    assert is_api_endpoint(path) == expected


@pytest.mark.parametrize(
    ("origin", "expect_cors_header"),
    [
        ("http://localhost:3000", True),
        ("http://127.0.0.1:5000", True),
        ("http://[::1]:8080", True),
        ("http://evil.com", False),
    ],
)
def test_fastapi_cors_allows_localhost_origins(fastapi_client, origin, expect_cors_header):
    response = fastapi_client.get(
        "/api/2.0/mlflow/experiments/list", headers={"Host": "localhost", "Origin": origin}
    )
    if expect_cors_header:
        assert response.headers.get("access-control-allow-origin") == origin
    else:
        assert response.headers.get("access-control-allow-origin") is None


def test_fastapi_wildcard_cors_disables_credentials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "*")
    security_logger = logging.getLogger("mlflow.server.fastapi_security")
    security_logger.addHandler(caplog.handler)

    app = FastAPI()

    @app.api_route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    async def api_endpoint():
        return {"ok": True}

    try:
        with caplog.at_level("WARNING", logger="mlflow.server.fastapi_security"):
            init_fastapi_security(app)
    finally:
        security_logger.removeHandler(caplog.handler)
    assert any("disabling credentialed CORS" in record.message for record in caplog.records)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/2.0/mlflow/experiments/list",
        headers={"Host": "localhost", "Origin": "http://evil.com"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-credentials") != "true"

    preflight = client.options(
        "/api/2.0/mlflow/experiments/list",
        headers={
            "Host": "localhost",
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.headers.get("access-control-allow-credentials") != "true"


def test_fastapi_cors_allows_configured_origin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", "https://trusted.com")

    app = FastAPI()

    @app.api_route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    async def api_endpoint():
        return {"ok": True}

    init_fastapi_security(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/api/2.0/mlflow/experiments/list",
        headers={"Host": "localhost", "Origin": "https://trusted.com"},
    )
    assert response.headers.get("access-control-allow-origin") == "https://trusted.com"

    response = client.get(
        "/api/2.0/mlflow/experiments/list",
        headers={"Host": "localhost", "Origin": "http://evil.com"},
    )
    assert response.headers.get("access-control-allow-origin") is None
