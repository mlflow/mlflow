import time
from dataclasses import dataclass
from unittest import mock

import pytest
import requests

from mlflow.server.auth.oauth.external_authz import ExternalAuthzClient


@dataclass
class FakeAuthzConfig:
    enabled: bool = True
    endpoint: str = "https://authz.example.com/v1/check"
    forward_token: bool = True
    headers: str = ""
    allowed_field: str = "allowed"
    permission_field: str = "permission"
    admin_field: str = "is_admin"
    cache_ttl_seconds: int = 300
    cache_max_size: int = 100
    timeout_seconds: int = 5
    max_retries: int = 1
    retry_backoff_seconds: float = 0.0
    on_error: str = "deny"


@pytest.fixture
def config():
    return FakeAuthzConfig()


@pytest.fixture
def client(config):
    return ExternalAuthzClient(config)


def test_external_authz_returns_none_when_disabled():
    config = FakeAuthzConfig(enabled=False)
    client = ExternalAuthzClient(config)
    result = client.check_permission("user", "", "", "experiment", "1", "read")
    assert result is None


def test_external_authz_allow_response(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "EDIT",
        "is_admin": False,
        "cache_ttl_seconds": 60,
    }

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        result = client.check_permission(
            username="jane",
            email="jane@example.com",
            provider="oidc:primary",
            resource_type="experiment",
            resource_id="123",
            action="read",
            access_token="token123",
            ip_address="10.0.0.1",
        )
        mock_post.assert_called_once()

    assert result["allowed"] is True
    assert result["permission"] == "EDIT"
    assert result["is_admin"] is False


def test_external_authz_deny_response(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": False,
        "permission": "NO_PERMISSIONS",
        "is_admin": False,
        "reason": "Not in group",
    }

    with mock.patch("requests.post", return_value=mock_resp):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result["allowed"] is False
    assert result["reason"] == "Not in group"


def test_external_authz_404_falls_through(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 404

    with mock.patch("requests.post", return_value=mock_resp):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result is None


def test_external_authz_on_error_deny(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 500

    with mock.patch("requests.post", return_value=mock_resp):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result["allowed"] is False


def test_external_authz_on_error_fallback_to_default():
    config = FakeAuthzConfig(on_error="fallback_to_default")
    client = ExternalAuthzClient(config)

    mock_resp = mock.Mock()
    mock_resp.status_code = 500

    with mock.patch("requests.post", return_value=mock_resp):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result is None


def test_external_authz_on_error_allow():
    config = FakeAuthzConfig(on_error="allow")
    client = ExternalAuthzClient(config)

    mock_resp = mock.Mock()
    mock_resp.status_code = 500

    with mock.patch("requests.post", return_value=mock_resp):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result["allowed"] is True


def test_external_authz_timeout_triggers_on_error(client):
    with mock.patch("requests.post", side_effect=requests.exceptions.Timeout):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result["allowed"] is False


def test_external_authz_cache_hit(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "READ",
        "is_admin": False,
        "cache_ttl_seconds": 300,
    }

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        result = client.check_permission("jane", "", "", "experiment", "1", "read")
        assert mock_post.call_count == 1

    assert result["allowed"] is True


def test_external_authz_cache_ttl_expiry():
    config = FakeAuthzConfig(cache_ttl_seconds=300)
    client = ExternalAuthzClient(config)

    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "READ",
        "is_admin": False,
        "cache_ttl_seconds": 0.1,
    }

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        assert mock_post.call_count == 1

    time.sleep(0.15)

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        mock_post.assert_called_once()


def test_external_authz_cache_invalidation_by_user(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "READ",
        "is_admin": False,
        "cache_ttl_seconds": 300,
    }

    with mock.patch("requests.post", return_value=mock_resp):
        client.check_permission("jane", "", "", "experiment", "1", "read")

    client.invalidate_cache_for_user("jane")

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        mock_post.assert_called_once()


def test_external_authz_cache_invalidation_by_resource(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "READ",
        "is_admin": False,
        "cache_ttl_seconds": 300,
    }

    with mock.patch("requests.post", return_value=mock_resp):
        client.check_permission("jane", "", "", "experiment", "1", "read")

    client.invalidate_cache_for_resource("experiment", "1")

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        mock_post.assert_called_once()


def test_external_authz_request_payload_format(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"allowed": True, "cache_ttl_seconds": 0}

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission(
            username="jane",
            email="jane@example.com",
            provider="oidc:primary",
            resource_type="experiment",
            resource_id="123",
            action="read",
            access_token="mytoken",
            ip_address="10.0.0.1",
            workspace="prod",
        )
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]

        assert payload["subject"]["username"] == "jane"
        assert payload["subject"]["email"] == "jane@example.com"
        assert payload["subject"]["provider"] == "oidc:primary"
        assert payload["resource"]["type"] == "experiment"
        assert payload["resource"]["id"] == "123"
        assert payload["resource"]["workspace"] == "prod"
        assert payload["action"] == "read"
        assert payload["context"]["ip_address"] == "10.0.0.1"
        assert "timestamp" in payload["context"]

        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer mytoken"
        assert headers["X-MLflow-Service"] == "mlflow"


def test_external_authz_custom_headers():
    config = FakeAuthzConfig(headers="X-Custom: val1, X-Env: prod")
    client = ExternalAuthzClient(config)

    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"allowed": True, "cache_ttl_seconds": 0}

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Custom"] == "val1"
        assert headers["X-Env"] == "prod"


def test_external_authz_retry_on_transient_error():
    config = FakeAuthzConfig(max_retries=2, retry_backoff_seconds=0.0)
    client = ExternalAuthzClient(config)

    error_resp = mock.Mock()
    error_resp.status_code = 503

    ok_resp = mock.Mock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"allowed": True, "cache_ttl_seconds": 0}

    with mock.patch("requests.post", side_effect=[error_resp, error_resp, ok_resp]):
        result = client.check_permission("jane", "", "", "experiment", "1", "read")

    assert result["allowed"] is True


def test_external_authz_auth_failure_no_retry():
    config = FakeAuthzConfig(max_retries=2, retry_backoff_seconds=0.0)
    client = ExternalAuthzClient(config)

    mock_resp = mock.Mock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        result = client.check_permission("jane", "", "", "experiment", "1", "read")
        mock_post.assert_called_once()

    assert result["allowed"] is False


def test_external_authz_clear_cache(client):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "allowed": True,
        "permission": "READ",
        "is_admin": False,
        "cache_ttl_seconds": 300,
    }

    with mock.patch("requests.post", return_value=mock_resp):
        client.check_permission("jane", "", "", "experiment", "1", "read")

    client.clear_cache()

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client.check_permission("jane", "", "", "experiment", "1", "read")
        mock_post.assert_called_once()
