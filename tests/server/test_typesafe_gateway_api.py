from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import mlflow
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.gateway.config import GatewayRequestType
from mlflow.gateway.guardrails import GuardrailViolation
from mlflow.server.gateway_api import gateway_router, typesafe_passthrough_system_one
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig, GatewayModelConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture(params=[None, "team-a"])
def endpoint(request, tmp_path, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "typesafe-gateway-test-passphrase")
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(request.param is not None).lower())
    store_class = WorkspaceAwareSqlAlchemyStore if request.param else SqlAlchemyStore
    store = store_class(db_uri, tmp_path.as_uri())
    mlflow.set_tracking_uri(db_uri)
    with WorkspaceContext(request.param):
        experiment_id = store.create_experiment("gateway/jev-evaluator")
        secret = store.create_gateway_secret(
            secret_name="typesafe-key",
            secret_value={"api_key": "typesafe-secret"},
            provider="typesafe",
        )
        model = store.create_gateway_model_definition(
            name="jev-model",
            secret_id=secret.secret_id,
            provider="typesafe",
            model_name="jev-1.13.0",
        )
        endpoint = store.create_gateway_endpoint(
            name="jev-evaluator",
            model_configs=[
                GatewayEndpointModelConfig(
                    model_definition_id=model.model_definition_id,
                    linkage_type=GatewayModelLinkageType.PRIMARY,
                    weight=1.0,
                )
            ],
            usage_tracking=True,
            experiment_id=experiment_id,
        )
        with patch("mlflow.server.gateway_api._get_store", return_value=store):
            yield SimpleNamespace(
                store=store,
                endpoint=endpoint,
                experiment_id=experiment_id,
                workspace=request.param,
            )
    mlflow.set_tracking_uri(None)


def _request():
    return {
        "model": "jev-evaluator",
        "state": {"inputs": "What is MLflow?", "outputs": "An ML platform."},
        "questions": {"relevance": {"type": "noul", "instructions": "Is the answer relevant?"}},
    }


def _response():
    return {
        "model": "jev-1.13.0",
        "answers": {"relevance": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_system_one_route_credentials_tracing_and_budgets(endpoint):
    app = FastAPI()
    app.include_router(gateway_router)
    client = TestClient(app)
    with (
        patch("mlflow.gateway.providers.typesafe.send_request", return_value=_response()) as send,
        patch("mlflow.server.gateway_api.check_budget_limit") as budget,
    ):
        response = client.post("/gateway/typesafe/v1/systemone", json=_request())

    assert response.status_code == 200
    assert response.json() == _response()
    send.assert_awaited_once_with(
        headers={"Authorization": "Bearer typesafe-secret"},
        base_url="https://api.typesafe.ai/v1",
        path="systemone",
        payload={**_request(), "model": "jev-1.13.0"},
    )
    assert budget.call_args.kwargs == {"workspace": endpoint.workspace, "username": None}
    assert budget.call_args.args[1].endpoint_id == endpoint.endpoint.endpoint_id

    traces = TracingClient().search_traces(locations=[endpoint.experiment_id])
    assert len(traces) == 1
    trace = traces[0]
    assert (
        trace.info.request_metadata[TraceMetadataKey.GATEWAY_ENDPOINT_ID]
        == endpoint.endpoint.endpoint_id
    )
    assert (
        trace.info.request_metadata[TraceMetadataKey.GATEWAY_REQUEST_TYPE]
        == GatewayRequestType.PASSTHROUGH_MODEL_TYPESAFE_SYSTEM_ONE
    )
    span = next(span for span in trace.data.spans if "action" in span.attributes)
    assert span.attributes[SpanAttributeKey.MODEL_PROVIDER] == "typesafe"
    assert span.attributes[SpanAttributeKey.CHAT_USAGE] == {
        TokenUsageKey.INPUT_TOKENS: 100,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 120,
    }
    assert span.outputs == _response()


@pytest.mark.parametrize("endpoint", ["team-a"], indirect=True)
def test_system_one_cannot_invoke_endpoint_in_another_workspace(endpoint):
    app = FastAPI()
    app.include_router(gateway_router)
    with (
        WorkspaceContext("team-b"),
        patch("mlflow.gateway.providers.typesafe.send_request") as send,
    ):
        response = TestClient(app).post("/gateway/typesafe/v1/systemone", json=_request())
    assert response.status_code == 404
    send.assert_not_called()


@pytest.mark.asyncio
async def test_system_one_guardrails_and_user_budget(endpoint):
    request = MagicMock()
    request.state.cached_body = _request()
    request.state.username = "alice"
    request.state.user_id = None
    request.headers = {"authorization": "Basic mlflow-auth"}
    pre_guardrail = MagicMock()
    post_guardrail = MagicMock()
    modified_body = {**_request(), "state": "Sanitized input"}
    modified_body.pop("model")
    with (
        patch("mlflow.gateway.providers.typesafe.send_request", return_value=_response()) as send,
        patch("mlflow.server.gateway_api.check_budget_limit") as budget,
        patch(
            "mlflow.server.gateway_api._get_guardrails_and_auth",
            return_value=([pre_guardrail, post_guardrail], request.headers),
        ),
        patch(
            "mlflow.server.gateway_api.run_pre_llm_guardrails", return_value=modified_body
        ) as pre,
        patch(
            "mlflow.server.gateway_api.run_post_llm_guardrails_passthrough",
            return_value=_response(),
        ) as post,
    ):
        assert await typesafe_passthrough_system_one(request) == _response()

    assert send.call_args.kwargs["payload"]["state"] == "Sanitized input"
    assert budget.call_args.kwargs == {"workspace": endpoint.workspace, "username": "alice"}
    assert pre.call_args.args[0] == [pre_guardrail, post_guardrail]
    assert post.call_args.args == ([pre_guardrail, post_guardrail], modified_body, _response())
    assert post.call_args.kwargs["auth_headers"] == request.headers


@pytest.mark.parametrize("failure", ["budget", "guardrail"])
@pytest.mark.asyncio
async def test_system_one_stops_before_upstream_on_policy_failure(endpoint, failure):
    request = MagicMock()
    request.state.cached_body = _request()
    request.state.username = None
    request.state.user_id = None
    with patch("mlflow.gateway.providers.typesafe.send_request") as send:
        if failure == "budget":
            with patch(
                "mlflow.server.gateway_api.check_budget_limit",
                side_effect=HTTPException(429, "Budget exceeded"),
            ):
                with pytest.raises(HTTPException, match="Budget exceeded") as exc:
                    await typesafe_passthrough_system_one(request)
            assert exc.value.status_code == 429
        else:
            with patch(
                "mlflow.server.gateway_api.run_pre_llm_guardrails",
                side_effect=GuardrailViolation("test-guardrail", "Blocked"),
            ):
                with pytest.raises(HTTPException, match="Blocked") as exc:
                    await typesafe_passthrough_system_one(request)
            assert exc.value.status_code == 400
        send.assert_not_called()


@pytest.mark.parametrize(
    "linkage", [GatewayModelLinkageType.PRIMARY, GatewayModelLinkageType.FALLBACK]
)
@pytest.mark.asyncio
async def test_system_one_rejects_mixed_model_providers(linkage):
    request = MagicMock()
    request.state.cached_body = _request()
    request.state.username = None
    request.state.user_id = None
    provider = MagicMock()
    provider.passthrough = AsyncMock()
    config = GatewayEndpointConfig(
        endpoint_id="ep-test",
        endpoint_name="jev-evaluator",
        models=[
            GatewayModelConfig("md-jev", "typesafe", "jev-latest", {"api_key": "key"}),
            GatewayModelConfig(
                "md-chat", "openai", "gpt-4o", {"api_key": "key"}, linkage_type=linkage
            ),
        ],
    )
    with (
        patch("mlflow.server.gateway_api._validate_store"),
        patch(
            "mlflow.server.gateway_api._create_provider_from_endpoint_name",
            return_value=(provider, config),
        ),
    ):
        with pytest.raises(HTTPException, match="requires all endpoint models") as exc:
            await typesafe_passthrough_system_one(request)
    assert exc.value.status_code == 400
    provider.passthrough.assert_not_called()


@pytest.mark.parametrize("body", [{"state": "Example"}, {**_request(), "stream": True}])
def test_system_one_rejects_invalid_request_before_endpoint_lookup(body):
    app = FastAPI()
    app.include_router(gateway_router)
    with patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as create_provider:
        response = TestClient(app).post("/gateway/typesafe/v1/systemone", json=body)
    assert response.status_code == 400
    create_provider.assert_not_called()


def test_system_one_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_AI_GATEWAY", "false")
    app = FastAPI()
    app.include_router(gateway_router)
    assert (
        TestClient(app).post("/gateway/typesafe/v1/systemone", json=_request()).status_code == 501
    )
