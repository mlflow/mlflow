from unittest import mock

import pytest
from fastapi import FastAPI
from flask import Response as FlaskResponse
from starlette.responses import Response as StarletteResponse
from starlette.testclient import TestClient
from werkzeug.datastructures import Authorization

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode
from mlflow.server.auth import (
    DEFAULT_AUTHORIZATION_FUNCTION,
    _authenticate_custom_for_fastapi,
    _flask_response_to_starlette,
    add_fastapi_permission_middleware,
)


def test_flask_response_to_starlette_preserves_status_code():
    flask_resp = FlaskResponse("Forbidden", status=403)
    result = _flask_response_to_starlette(flask_resp)
    assert isinstance(result, StarletteResponse)
    assert result.status_code == 403


def test_flask_response_to_starlette_preserves_body():
    flask_resp = FlaskResponse("Custom error message", status=401)
    result = _flask_response_to_starlette(flask_resp)
    assert result.body == b"Custom error message"


def test_flask_response_to_starlette_preserves_custom_headers():
    flask_resp = FlaskResponse("Unauthorized", status=401)
    flask_resp.headers["WWW-Authenticate"] = 'Bearer realm="mlflow"'
    flask_resp.headers["X-Custom-Header"] = "custom-value"
    result = _flask_response_to_starlette(flask_resp)
    assert result.headers["www-authenticate"] == 'Bearer realm="mlflow"'
    assert result.headers["x-custom-header"] == "custom-value"


def test_flask_response_to_starlette_excludes_hop_by_hop_headers():
    flask_resp = FlaskResponse("OK", status=200)
    flask_resp.headers["Transfer-Encoding"] = "chunked"
    result = _flask_response_to_starlette(flask_resp)
    assert "transfer-encoding" not in result.headers


def test_flask_response_to_starlette_preserves_content_encoding():
    flask_resp = FlaskResponse("OK", status=200)
    flask_resp.headers["Content-Encoding"] = "gzip"
    result = _flask_response_to_starlette(flask_resp)
    assert result.headers["content-encoding"] == "gzip"


def test_flask_response_to_starlette_preserves_multi_value_headers():
    flask_resp = FlaskResponse("OK", status=200)
    flask_resp.headers.add("Set-Cookie", "session=abc; Path=/")
    flask_resp.headers.add("Set-Cookie", "token=xyz; Path=/api")
    result = _flask_response_to_starlette(flask_resp)
    cookie_values = result.headers.getlist("set-cookie")
    assert len(cookie_values) == 2
    assert "session=abc; Path=/" in cookie_values
    assert "token=xyz; Path=/api" in cookie_values


@pytest.fixture
def mock_starlette_request():
    request = mock.MagicMock()
    request.url.path = "/api/2.0/mlflow-artifacts/artifacts/model.pkl"
    request.url.query = "run_id=abc123"
    request.method = "GET"
    request.headers = {"Authorization": "Bearer mytoken", "Host": "localhost"}
    return request


def test_custom_auth_returns_user_on_successful_auth(mock_starlette_request):
    mock_user = mock.MagicMock()
    mock_user.username = "testuser"
    mock_authorization = Authorization("basic", {"username": "testuser"})

    with (
        mock.patch(
            "mlflow.server.auth.authenticate_request",
            return_value=mock_authorization,
        ),
        mock.patch("mlflow.server.auth.store") as mock_store,
    ):
        mock_store.get_user.return_value = mock_user
        result = _authenticate_custom_for_fastapi(mock_starlette_request)

    assert result is mock_user
    mock_store.get_user.assert_called_once_with("testuser")


def test_custom_auth_returns_starlette_response_on_flask_response(mock_starlette_request):
    flask_resp = FlaskResponse("Auth failed", status=401)
    flask_resp.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'

    with mock.patch(
        "mlflow.server.auth.authenticate_request",
        return_value=flask_resp,
    ):
        result = _authenticate_custom_for_fastapi(mock_starlette_request)

    assert isinstance(result, StarletteResponse)
    assert result.status_code == 401
    assert result.body == b"Auth failed"


def test_custom_auth_raises_on_unsupported_authorization_result(mock_starlette_request):
    with mock.patch(
        "mlflow.server.auth.authenticate_request",
        return_value="unexpected_type",
    ):
        with pytest.raises(MlflowException, match="Unsupported result type") as exc_info:
            _authenticate_custom_for_fastapi(mock_starlette_request)

    assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_fastapi_middleware_returns_json_for_unsupported_custom_auth():
    app = FastAPI()
    add_fastapi_permission_middleware(app)

    @app.get("/api/2.0/mlflow-artifacts/artifacts/{artifact_path:path}")
    async def _download(artifact_path: str):
        return {"ok": True}

    mock_config = mock.Mock()
    mock_config.authorization_function = "tests.server.auth.fake_auth:broken_auth"
    with (
        mock.patch("mlflow.server.auth.auth_config", mock_config),
        mock.patch(
            "mlflow.server.auth.authenticate_request",
            return_value="unexpected_type",
        ),
    ):
        assert mock_config.authorization_function != DEFAULT_AUTHORIZATION_FUNCTION
        response = TestClient(app, raise_server_exceptions=False).get(
            "/api/2.0/mlflow-artifacts/artifacts/1/run/artifacts/model.pkl"
        )

    assert response.status_code == 500
    assert response.headers.get("www-authenticate") is None
    body = response.json()
    assert body["error_code"] == ErrorCode.Name(INTERNAL_ERROR)
    assert "Unsupported result type" in body["message"]


def test_custom_auth_returns_none_on_missing_username(mock_starlette_request):
    mock_authorization = Authorization("basic", {"username": None})

    with mock.patch(
        "mlflow.server.auth.authenticate_request",
        return_value=mock_authorization,
    ):
        result = _authenticate_custom_for_fastapi(mock_starlette_request)

    assert result is None


def test_custom_auth_returns_none_on_user_not_found(mock_starlette_request):
    mock_authorization = Authorization("basic", {"username": "ghost"})

    with (
        mock.patch(
            "mlflow.server.auth.authenticate_request",
            return_value=mock_authorization,
        ),
        mock.patch("mlflow.server.auth.store") as mock_store,
    ):
        mock_store.get_user.side_effect = Exception("User not found")
        result = _authenticate_custom_for_fastapi(mock_starlette_request)

    assert result is None


def test_custom_auth_passes_request_context_correctly(mock_starlette_request):
    mock_authorization = Authorization("basic", {"username": "testuser"})
    captured_path = None
    captured_method = None

    def capture_context():
        from flask import request as flask_request

        nonlocal captured_path, captured_method
        captured_path = flask_request.path
        captured_method = flask_request.method
        return mock_authorization

    mock_user = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.server.auth.authenticate_request",
            side_effect=capture_context,
        ),
        mock.patch("mlflow.server.auth.store") as mock_store,
    ):
        mock_store.get_user.return_value = mock_user
        _authenticate_custom_for_fastapi(mock_starlette_request)

    assert captured_path == "/api/2.0/mlflow-artifacts/artifacts/model.pkl"
    assert captured_method == "GET"
