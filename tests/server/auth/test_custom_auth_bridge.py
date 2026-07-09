"""Tests for the custom authorization_function bridge to FastAPI.

Verifies that custom auth functions (written against Flask's request context) work
correctly when invoked through the FastAPI permission middleware via the bridge in
``_authenticate_custom_for_fastapi``.
"""

from unittest import mock

import pytest
from flask import Response as FlaskResponse
from starlette.responses import Response as StarletteResponse
from werkzeug.datastructures import Authorization

from mlflow.server.auth import (
    _authenticate_custom_for_fastapi,
    _flask_response_to_starlette,
)


class TestFlaskResponseToStarlette:
    def test_preserves_status_code(self):
        flask_resp = FlaskResponse("Forbidden", status=403)
        result = _flask_response_to_starlette(flask_resp)
        assert isinstance(result, StarletteResponse)
        assert result.status_code == 403

    def test_preserves_body(self):
        flask_resp = FlaskResponse("Custom error message", status=401)
        result = _flask_response_to_starlette(flask_resp)
        assert result.body == b"Custom error message"

    def test_preserves_custom_headers(self):
        flask_resp = FlaskResponse("Unauthorized", status=401)
        flask_resp.headers["WWW-Authenticate"] = 'Bearer realm="mlflow"'
        flask_resp.headers["X-Custom-Header"] = "custom-value"
        result = _flask_response_to_starlette(flask_resp)
        assert result.headers["www-authenticate"] == 'Bearer realm="mlflow"'
        assert result.headers["x-custom-header"] == "custom-value"

    def test_excludes_hop_by_hop_headers(self):
        flask_resp = FlaskResponse("OK", status=200)
        flask_resp.headers["Transfer-Encoding"] = "chunked"
        result = _flask_response_to_starlette(flask_resp)
        assert "transfer-encoding" not in result.headers

    def test_preserves_content_encoding(self):
        flask_resp = FlaskResponse("OK", status=200)
        flask_resp.headers["Content-Encoding"] = "gzip"
        result = _flask_response_to_starlette(flask_resp)
        assert result.headers["content-encoding"] == "gzip"

    def test_preserves_multi_value_headers(self):
        flask_resp = FlaskResponse("OK", status=200)
        flask_resp.headers.add("Set-Cookie", "session=abc; Path=/")
        flask_resp.headers.add("Set-Cookie", "token=xyz; Path=/api")
        result = _flask_response_to_starlette(flask_resp)
        cookie_values = result.headers.getlist("set-cookie")
        assert len(cookie_values) == 2
        assert "session=abc; Path=/" in cookie_values
        assert "token=xyz; Path=/api" in cookie_values


class TestAuthenticateCustomForFastapi:
    @pytest.fixture
    def mock_starlette_request(self):
        request = mock.MagicMock()
        request.url.path = "/api/2.0/mlflow-artifacts/artifacts/model.pkl"
        request.url.query = "run_id=abc123"
        request.method = "GET"
        request.headers = {"Authorization": "Bearer mytoken", "Host": "localhost"}
        return request

    def test_returns_user_on_successful_auth(self, mock_starlette_request):
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

    def test_returns_starlette_response_on_flask_response(self, mock_starlette_request):
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

    def test_returns_none_on_non_authorization_result(self, mock_starlette_request):
        with mock.patch(
            "mlflow.server.auth.authenticate_request",
            return_value="unexpected_type",
        ):
            result = _authenticate_custom_for_fastapi(mock_starlette_request)

        assert result is None

    def test_returns_none_on_missing_username(self, mock_starlette_request):
        mock_authorization = Authorization("basic", {"username": None})

        with mock.patch(
            "mlflow.server.auth.authenticate_request",
            return_value=mock_authorization,
        ):
            result = _authenticate_custom_for_fastapi(mock_starlette_request)

        assert result is None

    def test_returns_none_on_user_not_found(self, mock_starlette_request):
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

    def test_passes_request_context_correctly(self, mock_starlette_request):
        """Ensure the bridge creates the Flask request context with correct path/method."""
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
