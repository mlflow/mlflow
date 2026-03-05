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


# -- Bearer token on gateway routes --


def test_bearer_valid_token_without_username_returns_none(
    mock_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


def test_bearer_valid_token_with_username_returns_that_user(
    mock_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123:alice")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.get_user.assert_called_once_with("alice")


def test_bearer_invalid_token_returns_none(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer wrong-token")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


def test_bearer_no_internal_token_configured_returns_none(
    mock_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, raising=False)
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


def test_bearer_deleted_user_returns_none(mock_store, mock_auth_config, monkeypatch):
    from mlflow.exceptions import MlflowException

    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    mock_store.get_user.side_effect = MlflowException("User not found")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123:deleted_user")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None


# -- Bearer token on non-gateway routes --


def test_bearer_rejected_on_non_gateway_path(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/api/3.0/mlflow/experiments/list", "Bearer abc123")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


def test_bearer_rejected_on_ajax_path(mock_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_INTERNAL_GATEWAY_AUTH_TOKEN_ENV_VAR, "abc123")
    request = _make_request("/ajax-api/3.0/jobs", "Bearer abc123")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_store.get_user.assert_not_called()


# -- Basic auth --


def test_valid_basic_auth(mock_store, mock_auth_config, monkeypatch):
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_invalid_basic_auth(mock_store, mock_auth_config, monkeypatch):
    mock_store.authenticate_user.return_value = False
    credentials = base64.b64encode(b"alice:wrong").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    from mlflow.server.auth import _authenticate_fastapi_request

    user = _authenticate_fastapi_request(request)

    assert user is None


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
