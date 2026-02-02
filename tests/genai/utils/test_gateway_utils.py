from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig, get_gateway_litellm_config


@pytest.mark.parametrize(
    ("gateway_uri", "tracking_uri", "endpoint_name", "expected_api_base"),
    [
        # MLFLOW_GATEWAY_URI set
        (
            "http://localhost:5000",
            "http://other:8000",
            "chat",
            "http://localhost:5000/gateway/mlflow/v1/",
        ),
        # MLFLOW_GATEWAY_URI not set, falls back to tracking URI
        (
            None,
            "http://localhost:5000",
            "my-endpoint",
            "http://localhost:5000/gateway/mlflow/v1/",
        ),
        # HTTPS URI
        (
            "https://mlflow.example.com",
            None,
            "chat",
            "https://mlflow.example.com/gateway/mlflow/v1/",
        ),
    ],
)
def test_get_gateway_litellm_config(
    gateway_uri, tracking_uri, endpoint_name, expected_api_base, monkeypatch
):
    if gateway_uri:
        monkeypatch.setenv("MLFLOW_GATEWAY_URI", gateway_uri)
    else:
        monkeypatch.delenv("MLFLOW_GATEWAY_URI", raising=False)

    with mock.patch(
        "mlflow.genai.utils.gateway_utils.get_tracking_uri",
        return_value=tracking_uri or "http://default:5000",
    ):
        config = get_gateway_litellm_config(endpoint_name)

    assert isinstance(config, GatewayLiteLLMConfig)
    assert config.api_base == expected_api_base
    assert config.api_key == "mlflow-gateway-auth"
    assert config.model == f"openai/{endpoint_name}"


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "sqlite:///mlflow.db",
        "/path/to/mlflow",
        "databricks",
    ],
)
def test_get_gateway_litellm_config_invalid_uri(tracking_uri, monkeypatch):
    monkeypatch.delenv("MLFLOW_GATEWAY_URI", raising=False)

    with mock.patch("mlflow.genai.utils.gateway_utils.get_tracking_uri", return_value=tracking_uri):
        with pytest.raises(MlflowException, match="Gateway provider requires an HTTP"):
            get_gateway_litellm_config("chat")
