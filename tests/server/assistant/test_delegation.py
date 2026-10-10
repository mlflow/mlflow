from unittest import mock

import pytest

from mlflow.server.assistant.delegation import (
    mint_delegation_credential,
    verify_delegation_credential,
)

_SECRET = "test-signing-secret"


@pytest.fixture
def signing_key(monkeypatch):
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", _SECRET)


def test_mint_and_verify_round_trip(signing_key):
    credential = mint_delegation_credential("alice")
    assert credential is not None
    assert verify_delegation_credential(credential) == "alice"


def test_username_with_colon_round_trips(signing_key):
    # A username containing ':' must still parse (the signature is url-safe base64 and the expiry
    # is digits, so rsplit isolates them correctly).
    credential = mint_delegation_credential("workspace:alice")
    assert verify_delegation_credential(credential) == "workspace:alice"


def test_expired_credential_rejected(signing_key):
    assert verify_delegation_credential(mint_delegation_credential("alice", ttl_seconds=-1)) is None


def test_tampered_username_rejected(signing_key):
    credential = mint_delegation_credential("alice")
    _, expiry, sig = credential.rsplit(":", 2)
    # Reuse alice's signature under a different username; verification must reject it.
    assert verify_delegation_credential(f"v1:bob:{expiry}:{sig}") is None


def test_signature_from_a_different_key_rejected(monkeypatch):
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", "secret-a")
    credential = mint_delegation_credential("alice")
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", "secret-b")
    assert verify_delegation_credential(credential) is None


def test_no_signing_key_mints_and_verifies_nothing(monkeypatch):
    monkeypatch.delenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", raising=False)
    assert mint_delegation_credential("alice") is None
    assert verify_delegation_credential("v1:alice:99999999999999:sig") is None


def test_no_username_mints_nothing(signing_key):
    assert mint_delegation_credential("") is None


@pytest.mark.parametrize("bad", [None, "", "garbage", "v1:alice", "v2:alice:1:sig"])
def test_malformed_credential_rejected(signing_key, bad):
    assert verify_delegation_credential(bad) is None


def test_tool_subprocess_env_carries_a_credential_for_the_current_user(monkeypatch):
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", _SECRET)
    monkeypatch.setattr("mlflow.assistant.config.get_config_user", lambda: "alice")
    from mlflow.assistant.providers.tool_executor import _assistant_delegation_env
    from mlflow.tracking.request_auth.assistant_delegation_request_auth_provider import (
        ASSISTANT_DELEGATION_AUTH_NAME,
    )

    env = _assistant_delegation_env()
    assert env["MLFLOW_TRACKING_AUTH"] == ASSISTANT_DELEGATION_AUTH_NAME
    assert verify_delegation_credential(env["_MLFLOW_ASSISTANT_DELEGATION_TOKEN"]) == "alice"


def test_tool_subprocess_env_empty_without_an_identity(monkeypatch):
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", _SECRET)
    monkeypatch.setattr("mlflow.assistant.config.get_config_user", lambda: None)
    from mlflow.assistant.providers.tool_executor import _assistant_delegation_env

    assert _assistant_delegation_env() == {}


def test_client_auth_provider_attaches_the_header(monkeypatch):
    monkeypatch.setenv("_MLFLOW_ASSISTANT_DELEGATION_TOKEN", "v1:alice:1:sig")
    from mlflow.tracking.request_auth.assistant_delegation_request_auth_provider import (
        ASSISTANT_DELEGATION_AUTH_NAME,
        ASSISTANT_DELEGATION_HEADER,
        AssistantDelegationRequestAuthProvider,
    )

    provider = AssistantDelegationRequestAuthProvider()
    assert provider.get_name() == ASSISTANT_DELEGATION_AUTH_NAME
    request = mock.MagicMock()
    request.headers = {}
    provider.get_auth()(request)
    assert request.headers[ASSISTANT_DELEGATION_HEADER] == "v1:alice:1:sig"
