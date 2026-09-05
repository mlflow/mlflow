import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlflow.gateway.constants import GATEWAY_DISABLED_MESSAGE
from mlflow.server.gateway_api import gateway_router


def test_gateway_endpoints_return_501_when_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_AI_GATEWAY", "false")

    app = FastAPI()
    app.include_router(gateway_router)
    client = TestClient(app)

    response = client.post(
        "/gateway/test-endpoint/mlflow/invocations",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 501
    assert response.json()["detail"] == GATEWAY_DISABLED_MESSAGE

    response = client.post(
        "/gateway/mlflow/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 501
    assert response.json()["detail"] == GATEWAY_DISABLED_MESSAGE

    response = client.post(
        "/gateway/openai/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 501
    assert response.json()["detail"] == GATEWAY_DISABLED_MESSAGE

    response = client.post(
        "/gateway/openai/v1/embeddings",
        json={"model": "test", "input": "Hello"},
    )
    assert response.status_code == 501
    assert response.json()["detail"] == GATEWAY_DISABLED_MESSAGE

    response = client.post(
        "/gateway/anthropic/v1/messages",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 501
    assert response.json()["detail"] == GATEWAY_DISABLED_MESSAGE


def test_gateway_endpoints_pass_through_when_enabled():
    app = FastAPI()
    app.include_router(gateway_router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/gateway/test-endpoint/mlflow/invocations",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code != 501


@pytest.mark.parametrize(
    "handler_name",
    [
        "_create_gateway_secret",
        "_get_gateway_secret_info",
        "_update_gateway_secret",
        "_delete_gateway_secret",
        "_list_gateway_secrets",
        "_create_gateway_endpoint",
        "_get_gateway_endpoint",
        "_update_gateway_endpoint",
        "_delete_gateway_endpoint",
        "_list_gateway_endpoints",
        "_create_gateway_model_definition",
        "_get_gateway_model_definition",
        "_update_gateway_model_definition",
        "_delete_gateway_model_definition",
        "_list_gateway_model_definitions",
        "_attach_model_to_gateway_endpoint",
        "_detach_model_from_gateway_endpoint",
        "_create_gateway_endpoint_binding",
        "_delete_gateway_endpoint_binding",
        "_list_gateway_endpoint_bindings",
        "_set_gateway_endpoint_tag",
        "_delete_gateway_endpoint_tag",
        "_create_budget_policy",
        "_get_budget_policy",
        "_update_budget_policy",
        "_delete_budget_policy",
        "_list_budget_policies",
        "_list_budget_windows",
        "_create_gateway_guardrail",
        "_get_gateway_guardrail",
        "_delete_gateway_guardrail",
        "_list_gateway_guardrails",
        "_add_guardrail_to_endpoint",
        "_remove_guardrail_from_endpoint",
        "_list_endpoint_guardrail_configs",
        "_update_endpoint_guardrail_config",
        "_list_supported_providers",
        "_list_supported_models",
        "_get_provider_config",
        "_get_secrets_config",
    ],
)
def test_flask_gateway_handlers_return_501_when_disabled(monkeypatch, handler_name):
    monkeypatch.setenv("MLFLOW_ENABLE_AI_GATEWAY", "false")

    from flask import Flask

    from mlflow.server import handlers

    handler = getattr(handlers, handler_name)
    flask_app = Flask(__name__)

    with flask_app.app_context():
        response = handler()
        assert response.status_code == 501
        assert response.get_json()["error_code"] == "NOT_IMPLEMENTED"
        assert response.get_json()["message"] == GATEWAY_DISABLED_MESSAGE


def test_server_info_includes_features_enabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_AI_GATEWAY", "false")

    from flask import Flask

    from mlflow.server.handlers import _get_server_info

    flask_app = Flask(__name__)

    with flask_app.app_context():
        response = _get_server_info()
        data = response.get_json()
        assert "features_enabled" in data
        assert data["features_enabled"]["gateway"] is False


def test_server_info_gateway_enabled_by_default():
    from flask import Flask

    from mlflow.server.handlers import _get_server_info

    flask_app = Flask(__name__)

    with flask_app.app_context():
        response = _get_server_info()
        data = response.get_json()
        assert "features_enabled" in data
        assert data["features_enabled"]["gateway"] is True
