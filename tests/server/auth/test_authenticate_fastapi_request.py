import base64
from unittest import mock

import pytest

from mlflow.server.auth import _INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.server.auth.store") as mock_store:
        mock_store.get_user.side_effect = lambda username: mock.Mock(username=username)
        mock_store.authenticate_user.return_value = True
        yield mock_store


@pytest.fixture
def mock_auth_config():
    with mock.patch("mlflow.server.auth.auth_config") as mock_config:
        mock_config.admin_username = "admin"
        yield mock_config


def _make_request(path, authorization=None):
    request = mock.Mock()
    request.url.path = path
    request.headers = {}
    if authorization:
        request.headers["Authorization"] = authorization
    return request


# -- Basic auth with internal token (trusted internal requests) --


def test_basic_auth_with_internal_token_returns_user(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "internal-secret")
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.get_user.assert_called_once_with("alice")
    mock_store.authenticate_user.assert_not_called()


def test_basic_auth_with_internal_token_deleted_user_returns_none(
    mock_store, mock_auth_config, monkeypatch
):
    from mlflow.exceptions import MlflowException

    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "internal-secret")
    mock_store.get_user.side_effect = MlflowException("User not found")
    credentials = base64.b64encode(b"deleted_user:internal-secret").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None


def test_basic_auth_with_wrong_password_falls_through_to_authenticate(
    mock_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "internal-secret")
    credentials = base64.b64encode(b"alice:wrong-password").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.authenticate_user.assert_called_once_with("alice", "wrong-password")


def test_basic_auth_no_internal_token_uses_normal_auth(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.authenticate_user.assert_called_once_with("alice", "password123")


# -- Standard Basic auth --


def test_valid_basic_auth(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_invalid_basic_auth(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, raising=False)
    mock_store.authenticate_user.return_value = False
    credentials = base64.b64encode(b"alice:wrong").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None


# -- Non-Basic auth schemes --


def test_bearer_returns_none(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


# -- No auth header --


def test_no_authorization_header(mock_store, mock_auth_config):
    request = _make_request("/api/3.0/mlflow/experiments/list")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None


def test_malformed_authorization_header(mock_store, mock_auth_config):
    request = _make_request("/api/3.0/mlflow/experiments/list", "garbage")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
