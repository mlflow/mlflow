import base64
from types import SimpleNamespace
from unittest import mock

import pytest
from packaging.version import Version

from mlflow.assistant.config import set_config_user
from mlflow.assistant.providers import MlflowGatewayProvider, list_providers
from mlflow.assistant.providers.base import ProviderNotConfiguredError
from mlflow.assistant.providers.mlflow_gateway import set_server_url
from mlflow.gateway.constants import MLFLOW_GATEWAY_AUTH_HEADER
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


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


@pytest.fixture
def reset_server_url():
    set_server_url(None)
    yield
    set_server_url(None)


def test_chat_url_comes_from_the_bound_server_url_not_the_tracking_uri(reset_server_url):
    # The chat call carries the internal gateway token, so it goes to the server's own address;
    # the tracking URI derives from the client's Host header (GHSA-98p3-crmr-mxcx).
    set_server_url("http://127.0.0.1:5000")

    chat_url = MlflowGatewayProvider._build_chat_url(None, "http://10.attacker.example:8080")

    assert chat_url == "http://127.0.0.1:5000/gateway/mlflow/v1/chat/completions"


def test_chat_url_is_unresolved_without_a_bound_server_url(reset_server_url):
    assert MlflowGatewayProvider._build_chat_url(None, "http://10.attacker.example:8080") is None


def test_auth_headers_present_the_bound_host_header(reset_server_url):
    # The self-call reaches the server at its own address but presents the Host header of the
    # request that triggered it, which the server's Host allowlist has already admitted.
    set_server_url("http://127.0.0.1:5000", host_header="mlflow.internal:5000")

    assert _gateway_provider()._auth_headers(None)["Host"] == "mlflow.internal:5000"


def test_auth_headers_omit_host_header_without_a_bound_one(reset_server_url):
    set_server_url("http://127.0.0.1:5000")

    assert "Host" not in _gateway_provider()._auth_headers(None)


def test_request_kwargs_verify_tls_against_the_bound_host_name(reset_server_url):
    # On a TLS listener the HTTPS self-call goes to the server's address, whose certificate
    # names the host clients connect to; verify it against the inbound Host name.
    set_server_url("https://127.0.0.1:5000", host_header="mlflow.internal:5000")

    kwargs = _gateway_provider()._request_kwargs(
        "https://127.0.0.1:5000/gateway/mlflow/v1/chat/completions"
    )

    assert kwargs == {"server_hostname": "mlflow.internal"}


@pytest.mark.parametrize(
    ("url", "host_header"),
    [
        ("http://127.0.0.1:5000", "mlflow.internal:5000"),
        ("https://127.0.0.1:5000", None),
    ],
)
def test_request_kwargs_empty_for_plain_http_or_without_a_bound_host(
    reset_server_url, url, host_header
):
    set_server_url(url, host_header=host_header)

    assert _gateway_provider()._request_kwargs(f"{url}/gateway/mlflow/v1/chat/completions") == {}


def test_auth_headers_present_the_bound_workspace_header(reset_server_url):
    # The self-call carries the inbound request's workspace header, so the gateway resolves the
    # same (non-default) workspace instead of falling back to the default one.
    set_server_url("http://127.0.0.1:5000", workspace_header="team-a")

    assert _gateway_provider()._auth_headers(None)[WORKSPACE_HEADER_NAME] == "team-a"


def test_auth_headers_omit_workspace_header_without_a_bound_one(reset_server_url):
    set_server_url("http://127.0.0.1:5000", host_header="mlflow.internal:5000")

    assert WORKSPACE_HEADER_NAME not in _gateway_provider()._auth_headers(None)


def test_request_kwargs_reject_tls_self_call_on_old_aiohttp(monkeypatch, reset_server_url):
    # Before aiohttp 3.9 there is no way to verify the certificate against the inbound host name
    # while connecting to the server's address, so refuse clearly instead of failing the handshake.
    monkeypatch.setattr(
        "mlflow.assistant.providers.mlflow_gateway._AIOHTTP_VERSION", Version("3.8.6")
    )
    set_server_url("https://127.0.0.1:5000", host_header="mlflow.internal:5000")

    with pytest.raises(ProviderNotConfiguredError, match="aiohttp>=3.9"):
        _gateway_provider()._request_kwargs(
            "https://127.0.0.1:5000/gateway/mlflow/v1/chat/completions"
        )


def test_check_connection_raises_when_no_backend_probe():
    # The provider has no backend listing strategy, so check_connection
    # must surface that explicitly rather than silently returning OK —
    # otherwise the health endpoint would claim a successful probe that
    # never ran. The frontend talks to the gateway endpoints API directly
    # for verification.
    with pytest.raises(NotImplementedError, match="verified by the frontend"):
        _gateway_provider().check_connection()
