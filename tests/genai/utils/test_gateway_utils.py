import base64
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig, get_gateway_litellm_config
from mlflow.utils.credentials import MlflowCreds

GATEWAY_URI = "http://localhost:5000"


@pytest.fixture
def gateway_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_URI", GATEWAY_URI)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)


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

    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    with mock.patch(
        "mlflow.genai.utils.gateway_utils.get_tracking_uri",
        return_value=tracking_uri or "http://default:5000",
    ):
        config = get_gateway_litellm_config(endpoint_name)

    assert isinstance(config, GatewayLiteLLMConfig)
    assert config.api_base == expected_api_base
    assert config.api_key == "mlflow-gateway-auth"
    assert config.model == f"openai/{endpoint_name}"
    assert config.extra_headers is None


def test_get_gateway_litellm_config_with_tracking_credentials(gateway_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret123")

    config = get_gateway_litellm_config("chat")

    expected_encoded = base64.b64encode(b"alice:secret123").decode("ascii")
    assert config.extra_headers == {"Authorization": f"Basic {expected_encoded}"}
    assert config.api_key == "mlflow-gateway-auth"


def test_get_gateway_litellm_config_without_tracking_credentials(gateway_env):
    config = get_gateway_litellm_config("chat")

    assert config.extra_headers is None
    assert config.api_key == "mlflow-gateway-auth"


def test_get_gateway_litellm_config_username_only_no_headers(gateway_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")

    config = get_gateway_litellm_config("chat")

    assert config.extra_headers is None


def test_get_gateway_litellm_config_with_credentials_file(gateway_env):
    with mock.patch(
        "mlflow.genai.utils.gateway_utils.read_mlflow_creds",
        return_value=MlflowCreds(username="bob", password="file-password"),
    ):
        config = get_gateway_litellm_config("chat")

    expected_encoded = base64.b64encode(b"bob:file-password").decode("ascii")
    assert config.extra_headers == {"Authorization": f"Basic {expected_encoded}"}


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
