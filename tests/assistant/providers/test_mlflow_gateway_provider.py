"""Tests for the MLflow AI Gateway preset of OpenAICompatibleProvider.

The gateway preset has no backend model-listing strategy and derives its
chat URL from the MLflow tracking URI (the same server hosting the
assistant API).
"""

import pytest

from mlflow.assistant.providers import list_providers


def _gateway_provider():
    for p in list_providers():
        if p.name == "mlflow_gateway":
            return p
    raise AssertionError("mlflow_gateway provider not registered")


def test_provider_identity():
    p = _gateway_provider()
    assert p.name == "mlflow_gateway"
    assert p.display_name == "MLflow AI Gateway"
    assert p.is_available() is True


def test_list_models_is_not_implemented():
    # Listing is handled on the frontend via the existing gateway ajax API,
    # so the assistant backend exposes no listing strategy for this preset.
    with pytest.raises(NotImplementedError, match="Model listing is not supported"):
        _gateway_provider().list_models()


def test_check_connection_no_ops_without_backend_listing():
    # The provider has no backend listing strategy, so check_connection
    # cannot probe and must succeed without raising.
    _gateway_provider().check_connection()
