# Runtime fail-closed enforcement for the FastAPI permission middleware: a native
# FastAPI route with no validator is denied when MLFLOW_BASIC_AUTH_FAIL_CLOSED is on,
# while paths delegated to the mounted Flask app pass through untouched.

from fastapi import FastAPI
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from mlflow.server import auth as a


async def _flask_like(scope, receive, send):
    # Stand-in for the mounted Flask WSGI app.
    await PlainTextResponse("flask-ok")(scope, receive, send)


def _build_client(monkeypatch):
    # No route resolves a validator, and nothing is "unprotected", so the fail-closed
    # branch is what decides native routes.
    monkeypatch.setattr(a, "is_unprotected_route", lambda path: False)
    monkeypatch.setattr(a, "_find_fastapi_validator", lambda path, method: None)

    app = FastAPI()

    @app.get("/api/3.0/mlflow/native-new-feature")
    async def native():
        return PlainTextResponse("native-ok")

    app.mount("/flask", _flask_like)
    a.add_fastapi_permission_middleware(app)
    return TestClient(app)


def test_native_route_without_validator_denied_when_fail_closed(monkeypatch):
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    client = _build_client(monkeypatch)
    assert client.get("/api/3.0/mlflow/native-new-feature").status_code == 403


def test_native_route_allowed_when_flag_off(monkeypatch):
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "false")
    client = _build_client(monkeypatch)
    resp = client.get("/api/3.0/mlflow/native-new-feature")
    assert resp.status_code == 200
    assert resp.text == "native-ok"


def test_flask_delegated_path_passes_through_even_when_fail_closed(monkeypatch):
    # A path served only by the mounted Flask app is not a native route, so the
    # middleware must not deny it — Flask's _before_request authorizes it downstream.
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    client = _build_client(monkeypatch)
    resp = client.get("/flask/anything")
    assert resp.status_code == 200
    assert resp.text == "flask-ok"


def test_documented_marker_exempts_native_route(monkeypatch):
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    monkeypatch.setattr(a, "_KNOWN_UNGATED_FASTAPI_ROUTE_MARKERS", ("/mlflow/native-new-feature",))
    client = _build_client(monkeypatch)
    assert client.get("/api/3.0/mlflow/native-new-feature").status_code == 200
