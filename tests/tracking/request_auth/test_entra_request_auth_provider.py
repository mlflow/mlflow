from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("azure.identity")

import mlflow.tracking.request_auth.entra_request_auth_provider as _entra_auth
from mlflow.environment_variables import MLFLOW_ENTRA_ID_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.entra_request_auth_provider import (
    AUTHORIZATION_HEADER_NAME,
    EntraAuth,
    EntraRequestAuthProvider,
)

_SCOPE = "api://dummy-client-id/.default"


@pytest.fixture(autouse=True)
def _reset_credential():
    _entra_auth._credential = None
    yield
    _entra_auth._credential = None


@pytest.fixture
def scope_env(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENTRA_ID_SCOPE.name, _SCOPE)


def _make_request(headers=None):
    return SimpleNamespace(headers=headers if headers is not None else {})


def _mock_credential(token="dummy-token"):
    credential = mock.Mock()
    credential.get_token.return_value = SimpleNamespace(token=token)
    return credential


def test_provider_registration():
    provider = EntraRequestAuthProvider()
    assert provider.get_name() == "entra"
    assert isinstance(provider.get_auth(), EntraAuth)


def test_provider_raises_without_azure_identity():
    with (
        mock.patch.dict("sys.modules", {"azure.identity": None}),
        pytest.raises(MlflowException, match="azure-identity.*not installed"),
    ):
        EntraRequestAuthProvider().get_auth()


def test_auth_raises_without_scope(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENTRA_ID_SCOPE.name, raising=False)
    with pytest.raises(MlflowException, match=MLFLOW_ENTRA_ID_SCOPE.name):
        EntraAuth()(_make_request())


def test_auth_adds_bearer_token(scope_env):
    credential = _mock_credential(token="dummy-token")
    with mock.patch("azure.identity.DefaultAzureCredential", return_value=credential):
        request = EntraAuth()(_make_request())

    assert request.headers[AUTHORIZATION_HEADER_NAME] == "Bearer dummy-token"
    credential.get_token.assert_called_once_with(_SCOPE)


def test_auth_does_not_overwrite_existing_header(scope_env):
    credential = _mock_credential()
    with mock.patch("azure.identity.DefaultAzureCredential", return_value=credential):
        request = EntraAuth()(_make_request({AUTHORIZATION_HEADER_NAME: "Bearer preset"}))

    assert request.headers[AUTHORIZATION_HEADER_NAME] == "Bearer preset"
    credential.get_token.assert_not_called()


def test_auth_reuses_credential_across_requests(scope_env):
    credential = _mock_credential()
    with mock.patch(
        "azure.identity.DefaultAzureCredential", return_value=credential
    ) as credential_cls:
        auth = EntraAuth()
        auth(_make_request())
        auth(_make_request())

    credential_cls.assert_called_once()
    assert credential.get_token.call_count == 2


def test_auth_wraps_token_acquisition_failure(scope_env):
    credential = mock.Mock()
    credential.get_token.side_effect = Exception("no credentials available")
    with (
        mock.patch("azure.identity.DefaultAzureCredential", return_value=credential),
        pytest.raises(MlflowException, match=f"token for scope '{_SCOPE}'"),
    ):
        EntraAuth()(_make_request())
