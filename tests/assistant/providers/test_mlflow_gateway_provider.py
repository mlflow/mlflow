import pytest

from mlflow.assistant.providers import GATEWAY_PROVIDER_NAME, list_providers


def _gateway_provider():
    for p in list_providers():
        if p.name == GATEWAY_PROVIDER_NAME:
            return p
    raise AssertionError(f"{GATEWAY_PROVIDER_NAME} provider not registered")


def test_provider_identity():
    p = _gateway_provider()
    assert p.name == GATEWAY_PROVIDER_NAME
    assert p.display_name == "MLflow AI Gateway"
    assert p.is_available() is True


def test_list_models_is_not_implemented():
    # Listing is handled on the frontend via the existing gateway ajax API,
    # so the assistant backend exposes no listing strategy for this preset.
    with pytest.raises(NotImplementedError, match="Model listing is not supported"):
        _gateway_provider().list_models()


def test_check_connection_raises_when_no_backend_probe():
    # The provider has no backend listing strategy, so check_connection
    # must surface that explicitly rather than silently returning OK —
    # otherwise the health endpoint would claim a successful probe that
    # never ran. The frontend talks to the gateway endpoints API directly
    # for verification.
    with pytest.raises(NotImplementedError, match="verified by the frontend"):
        _gateway_provider().check_connection()
