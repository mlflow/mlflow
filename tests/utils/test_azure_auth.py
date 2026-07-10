import sys
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.azure_auth import _reset_credential_cache, get_azure_openai_token


class FakeClientAuthenticationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def make_token(token="entra-token", expires_in=3600):
    return SimpleNamespace(token=token, expires_on=time.time() + expires_in)


@pytest.fixture(autouse=True)
def reset_credential_cache():
    _reset_credential_cache()
    yield
    _reset_credential_cache()


@pytest.fixture
def fake_azure_identity():
    identity = mock.Mock()
    exceptions = mock.Mock(ClientAuthenticationError=FakeClientAuthenticationError)
    with mock.patch.dict(
        sys.modules,
        {
            "azure.identity": identity,
            "azure.core.exceptions": exceptions,
        },
    ):
        yield identity


def test_default_azure_credential_used_without_service_principal(fake_azure_identity):
    credential = mock.Mock(get_token=mock.Mock(return_value=make_token()))
    fake_azure_identity.DefaultAzureCredential.return_value = credential

    assert get_azure_openai_token() == "entra-token"

    fake_azure_identity.DefaultAzureCredential.assert_called_once_with()
    fake_azure_identity.ClientSecretCredential.assert_not_called()
    credential.get_token.assert_called_once_with("https://cognitiveservices.azure.com/.default")


def test_client_secret_credential_used_with_service_principal(fake_azure_identity):
    credential = mock.Mock(get_token=mock.Mock(return_value=make_token()))
    fake_azure_identity.ClientSecretCredential.return_value = credential

    token = get_azure_openai_token(
        client_id="client-id", tenant_id="tenant-id", client_secret="client-secret"
    )

    assert token == "entra-token"
    fake_azure_identity.ClientSecretCredential.assert_called_once_with(
        tenant_id="tenant-id", client_id="client-id", client_secret="client-secret"
    )
    fake_azure_identity.DefaultAzureCredential.assert_not_called()
    credential.get_token.assert_called_once_with("https://cognitiveservices.azure.com/.default")


def test_missing_azure_identity_raises_helpful_error():
    with (
        mock.patch.dict(sys.modules, {"azure.identity": None}),
        pytest.raises(MlflowException, match="pip install azure-identity"),
    ):
        get_azure_openai_token()


def test_credential_and_token_are_cached(fake_azure_identity):
    credential = mock.Mock(get_token=mock.Mock(return_value=make_token()))
    fake_azure_identity.DefaultAzureCredential.return_value = credential

    assert get_azure_openai_token() == "entra-token"
    assert get_azure_openai_token() == "entra-token"

    fake_azure_identity.DefaultAzureCredential.assert_called_once_with()
    credential.get_token.assert_called_once()


def test_token_is_refreshed_near_expiry(fake_azure_identity):
    expiring = make_token(token="old-token", expires_in=30)
    fresh = make_token(token="new-token")
    credential = mock.Mock(get_token=mock.Mock(side_effect=[expiring, fresh]))
    fake_azure_identity.DefaultAzureCredential.return_value = credential

    assert get_azure_openai_token() == "old-token"
    assert get_azure_openai_token() == "new-token"

    fake_azure_identity.DefaultAzureCredential.assert_called_once_with()
    assert credential.get_token.call_count == 2


def test_distinct_service_principals_use_distinct_credentials(fake_azure_identity):
    fake_azure_identity.ClientSecretCredential.side_effect = [
        mock.Mock(get_token=mock.Mock(return_value=make_token(token="token-a"))),
        mock.Mock(get_token=mock.Mock(return_value=make_token(token="token-b"))),
    ]

    token_a = get_azure_openai_token(
        client_id="client-a", tenant_id="tenant-id", client_secret="secret"
    )
    token_b = get_azure_openai_token(
        client_id="client-b", tenant_id="tenant-id", client_secret="secret"
    )

    assert token_a == "token-a"
    assert token_b == "token-b"
    assert fake_azure_identity.ClientSecretCredential.call_count == 2


def test_client_authentication_error_is_wrapped(fake_azure_identity):
    credential = mock.Mock(
        get_token=mock.Mock(side_effect=FakeClientAuthenticationError("bad credentials"))
    )
    fake_azure_identity.DefaultAzureCredential.return_value = credential

    with pytest.raises(MlflowException, match="bad credentials"):
        get_azure_openai_token()

    credential.get_token.assert_called_once()
