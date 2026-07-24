from unittest import mock

import pytest

from mlflow.assistant.providers import MlflowGatewayProvider, list_providers


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
    endpoint = mock.MagicMock()
    endpoint.name = "chat-endpoint"
    store = mock.MagicMock()
    store.list_gateway_endpoints.return_value = [endpoint]

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert _gateway_provider().list_models() == ["chat-endpoint"]

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


def test_check_connection_raises_when_no_backend_probe():
    # The provider has no backend listing strategy, so check_connection
    # must surface that explicitly rather than silently returning OK —
    # otherwise the health endpoint would claim a successful probe that
    # never ran. The frontend talks to the gateway endpoints API directly
    # for verification.
    with pytest.raises(NotImplementedError, match="verified by the frontend"):
        _gateway_provider().check_connection()
