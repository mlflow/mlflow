import base64
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.assistant.config import set_config_user
from mlflow.assistant.providers import MlflowGatewayProvider, list_providers
from mlflow.gateway.constants import MLFLOW_GATEWAY_AUTH_HEADER


def _gateway_provider():
    for p in list_providers():
        if p.name == MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
            return p
    raise AssertionError(f"{MlflowGatewayProvider.GATEWAY_PROVIDER_NAME} provider not registered")


def test_provider_identity():
    p = _gateway_provider()
    # Literal "mlflow_gateway" pins the wire-format contract: this value is
    # stored in user config files and mirrored by the frontend constant
    # GATEWAY_PROVIDER_ID, so changing it would break backwards compatibility.
    assert p.name == "mlflow_gateway"
    assert p.display_name == "MLflow AI Gateway"


def test_list_models_reads_gateway_endpoint_names():
    store = mock.MagicMock()
    store.list_gateway_endpoints.return_value = [
        SimpleNamespace(name="z-chat-endpoint"),
        SimpleNamespace(name=""),
        SimpleNamespace(name="a-chat-endpoint"),
    ]

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert _gateway_provider().list_models() == ["a-chat-endpoint", "z-chat-endpoint"]

    store.list_gateway_endpoints.assert_called_once()


def test_list_models_empty_without_gateway_store_support():
    store = mock.MagicMock()
    store.list_gateway_endpoints.side_effect = NotImplementedError("FileStore")

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert _gateway_provider().list_models() == []


@pytest.mark.parametrize(
    ("endpoint_names", "expected"),
    [
        (["chat-endpoint"], True),
        ([], False),
    ],
)
def test_is_available_reflects_gateway_endpoints(endpoint_names, expected):
    with mock.patch.object(MlflowGatewayProvider, "list_models", return_value=endpoint_names):
        assert _gateway_provider().is_available() is expected


@pytest.fixture
def reset_config_user():
    set_config_user(None)
    yield
    set_config_user(None)


def test_auth_headers_carry_internal_gateway_token_for_current_user(monkeypatch, reset_config_user):
    # On an auth-enabled server the in-server gateway requires credentials; the provider
    # authenticates with the server's internal gateway token, attributed to the current user.
    monkeypatch.setenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", "tok-123")
    set_config_user("alice")

    headers = _gateway_provider()._auth_headers(None)

    expected = "Basic " + base64.b64encode(b"alice:tok-123").decode()
    assert headers[MLFLOW_GATEWAY_AUTH_HEADER] == expected
    # The credential rides in the dedicated gateway header, never Authorization (which the
    # gateway forwards upstream to the LLM provider).
    assert "Authorization" not in headers


def test_auth_headers_omit_gateway_token_without_token(monkeypatch, reset_config_user):
    # A server with no auth generates no internal token, and the gateway needs no credential.
    monkeypatch.delenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", raising=False)
    set_config_user("alice")

    assert MLFLOW_GATEWAY_AUTH_HEADER not in _gateway_provider()._auth_headers(None)


def test_auth_headers_omit_gateway_token_without_user(monkeypatch, reset_config_user):
    # Without a resolved caller there is no identity to attribute the call to, so the token
    # is not sent (the call then fails closed rather than impersonating an arbitrary user).
    monkeypatch.setenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", "tok-123")
    set_config_user(None)

    assert MLFLOW_GATEWAY_AUTH_HEADER not in _gateway_provider()._auth_headers(None)


def test_check_connection_raises_when_no_backend_probe():
    # The provider has no backend listing strategy, so check_connection
    # must surface that explicitly rather than silently returning OK —
    # otherwise the health endpoint would claim a successful probe that
    # never ran. The frontend talks to the gateway endpoints API directly
    # for verification.
    with pytest.raises(NotImplementedError, match="verified by the frontend"):
        _gateway_provider().check_connection()
