import json
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mlflow
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.entities.gateway_guardrail import (
    GatewayGuardrail,
    GatewayGuardrailConfig,
    GuardrailAction,
    GuardrailStage,
)
from mlflow.entities.scorer import ScorerVersion
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail
from mlflow.gateway.schemas import chat
from mlflow.gateway.schemas.chat import ResponsePayload
from mlflow.server.gateway_api import (
    _load_guardrails,
    _run_after_guardrails,
    _run_before_guardrails,
    invocations,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

TEST_PASSPHRASE = "test-passphrase-for-guardrail-tests"


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", TEST_PASSPHRASE)


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(db_uri)
    yield SqlAlchemyStore(db_uri, artifact_uri.as_uri())
    mlflow.set_tracking_uri(None)


# ─── Shared helpers ──────────────────────────────────────────────────────────


class _SimpleScorer:
    """Minimal scorer that returns 'yes' or 'no' and tracks call count."""

    def __init__(self, *, passing: bool = True) -> None:
        self.call_count = 0
        self._passing = passing

    def __call__(self, **kwargs) -> str:
        self.call_count += 1
        return "yes" if self._passing else "no"


def _make_judge(stage, action=GuardrailAction.VALIDATION, *, passing=True):
    scorer = _SimpleScorer(passing=passing)
    return JudgeGuardrail(
        scorer=scorer,
        stage=GuardrailStage(stage),
        action=GuardrailAction(action),
        name=f"test-{stage.lower()}",
    )


def _make_request_payload():
    return {"messages": [{"role": "user", "content": "hello"}]}


def _make_response_payload():
    return ResponsePayload(**{
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    })


def _make_scorer_version(stage="BEFORE", action="VALIDATION"):
    return ScorerVersion(
        experiment_id="0",
        scorer_name="safety",
        scorer_version=1,
        serialized_scorer=json.dumps({
            "name": "safety",
            "builtin_scorer_class": "Safety",
            "instructions": "check safety",
        }),
        creation_time=0,
        scorer_id="s-1",
    )


def _make_guardrail_entity(stage="BEFORE", action="VALIDATION"):
    return GatewayGuardrail(
        guardrail_id="gr-1",
        name="safety",
        scorer=_make_scorer_version(stage, action),
        stage=GuardrailStage(stage),
        action=GuardrailAction(action),
        created_at=0,
        last_updated_at=0,
    )


def _make_guardrail_config(stage="BEFORE", action="VALIDATION"):
    return GatewayGuardrailConfig(
        endpoint_id="ep-1",
        guardrail_id="gr-1",
        execution_order=0,
        created_at=0,
        guardrail=_make_guardrail_entity(stage, action),
    )


def _make_mock_request(body: dict, headers: dict | None = None):
    req = MagicMock()
    req.state.cached_body = None
    req.state.username = None
    req.state.user_id = None
    req.json = AsyncMock(return_value=body)
    req.headers = headers or {}
    req.base_url = "http://localhost:5000/"
    return req


def _make_chat_response(content: str = "Hello!") -> chat.ResponsePayload:
    return chat.ResponsePayload(
        id="resp-id",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            chat.Choice(
                index=0,
                message=chat.ResponseMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=chat.ChatUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def _setup_endpoint(store: SqlAlchemyStore, name: str = "chat-endpoint"):
    secret = store.create_gateway_secret(
        secret_name=f"key-{name}",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name=f"model-{name}",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    return store.create_gateway_endpoint(
        name=name,
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            )
        ],
    )


# ─── _run_before_guardrails ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_before_guardrails_passes():
    g = _make_judge("BEFORE")
    payload = _make_request_payload()
    result = await _run_before_guardrails([g], payload)
    assert result == payload
    assert g.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_before_guardrails_skips_after_stage():
    g = _make_judge("AFTER")
    payload = _make_request_payload()
    result = await _run_before_guardrails([g], payload)
    assert result == payload
    assert g.scorer.call_count == 0


@pytest.mark.asyncio
async def test_run_before_guardrails_blocks():
    g = _make_judge("BEFORE", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked"):
        await _run_before_guardrails([g], _make_request_payload())


@pytest.mark.asyncio
async def test_run_before_guardrails_chains_multiple():
    g1 = _make_judge("BEFORE")
    g2 = _make_judge("BEFORE")
    payload = _make_request_payload()
    result = await _run_before_guardrails([g1, g2], payload)
    assert result == payload
    assert g1.scorer.call_count == 1
    assert g2.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_before_guardrails_stops_at_first_failure():
    g1 = _make_judge("BEFORE", passing=False)
    g2 = _make_judge("BEFORE")
    with pytest.raises(GuardrailViolation, match="blocked"):
        await _run_before_guardrails([g1, g2], _make_request_payload())
    assert g2.scorer.call_count == 0


# ─── _run_after_guardrails ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_after_guardrails_passes():
    g = _make_judge("AFTER")
    req = _make_request_payload()
    response = _make_response_payload()
    result = await _run_after_guardrails([g], req, response)
    assert result.choices[0].message.content == "hi there"
    assert g.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_after_guardrails_skips_before_stage():
    g = _make_judge("BEFORE")
    response = _make_response_payload()
    result = await _run_after_guardrails([g], _make_request_payload(), response)
    assert result is response
    assert g.scorer.call_count == 0


@pytest.mark.asyncio
async def test_run_after_guardrails_blocks():
    g = _make_judge("AFTER", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked"):
        await _run_after_guardrails([g], _make_request_payload(), _make_response_payload())


@pytest.mark.asyncio
async def test_run_after_guardrails_no_guardrails_returns_response():
    response = _make_response_payload()
    result = await _run_after_guardrails([], _make_request_payload(), response)
    assert result is response


# ─── _load_guardrails ─────────────────────────────────────────────────────────


def test_load_guardrails_empty():
    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = []
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    result = _load_guardrails(store, endpoint_config, request)
    assert result == []
    store.list_endpoint_guardrail_configs.assert_called_once_with("ep-1")


def test_load_guardrails_converts_real_entity():
    config = _make_guardrail_config(stage="BEFORE", action="VALIDATION")
    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = [config]
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    with mock.patch("mlflow.gateway.guardrails.JudgeGuardrail.from_entity") as mock_from_entity:
        mock_from_entity.return_value = _make_judge("BEFORE")
        result = _load_guardrails(store, endpoint_config, request)

    assert len(result) == 1
    assert isinstance(result[0], JudgeGuardrail)
    mock_from_entity.assert_called_once_with(config.guardrail, "http://localhost:5000")


def test_load_guardrails_skips_failed_conversion():
    good_config = _make_guardrail_config(stage="BEFORE")
    bad_config = _make_guardrail_config(stage="AFTER")
    bad_config.guardrail_id = "gr-bad"

    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = [bad_config, good_config]
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    good_judge = _make_judge("BEFORE")

    def from_entity_side_effect(entity, server_url):
        if entity is bad_config.guardrail:
            raise ValueError("bad scorer")
        return good_judge

    with mock.patch(
        "mlflow.gateway.guardrails.JudgeGuardrail.from_entity",
        side_effect=from_entity_side_effect,
    ):
        result = _load_guardrails(store, endpoint_config, request)

    assert result == [good_judge]


# ─── End-to-end invocations scenarios ────────────────────────────────────────


@pytest.mark.asyncio
async def test_invocations_before_guardrail_passes(store: SqlAlchemyStore):
    endpoint = _setup_endpoint(store, "ep-before-pass")
    mock_response = _make_chat_response("Safe response")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    passing_judge = _make_judge("BEFORE")

    with (
        patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create,
        patch("mlflow.server.gateway_api._load_guardrails", return_value=[passing_judge]),
    ):
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )

        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "Safe response"
    assert mock_provider.chat.called
    assert passing_judge.scorer.call_count == 1


@pytest.mark.asyncio
async def test_invocations_before_guardrail_blocks(store: SqlAlchemyStore):
    from fastapi import HTTPException

    endpoint = _setup_endpoint(store, "ep-before-block")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "bad input"}]})

    blocking_judge = _make_judge("BEFORE", passing=False)
    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock()

    with (
        patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create,
        patch("mlflow.server.gateway_api._load_guardrails", return_value=[blocking_judge]),
    ):
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )
        with pytest.raises(HTTPException, match="400"):
            await invocations(endpoint.name, mock_request)

    assert not mock_provider.chat.called
    assert blocking_judge.scorer.call_count == 1


@pytest.mark.asyncio
async def test_invocations_after_guardrail_passes(store: SqlAlchemyStore):
    endpoint = _setup_endpoint(store, "ep-after-pass")
    mock_response = _make_chat_response("Clean response")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    passing_judge = _make_judge("AFTER")

    with (
        patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create,
        patch("mlflow.server.gateway_api._load_guardrails", return_value=[passing_judge]),
    ):
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )

        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "Clean response"
    assert passing_judge.scorer.call_count == 1


@pytest.mark.asyncio
async def test_invocations_after_guardrail_blocks(store: SqlAlchemyStore):
    from fastapi import HTTPException

    endpoint = _setup_endpoint(store, "ep-after-block")
    mock_response = _make_chat_response("Unsafe output")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    blocking_judge = _make_judge("AFTER", passing=False)
    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(return_value=mock_response)

    with (
        patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create,
        patch("mlflow.server.gateway_api._load_guardrails", return_value=[blocking_judge]),
    ):
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )
        with pytest.raises(HTTPException, match="400"):
            await invocations(endpoint.name, mock_request)

    assert mock_provider.chat.called
    assert blocking_judge.scorer.call_count == 1


@pytest.mark.asyncio
async def test_invocations_bypass_header_skips_guardrails(store: SqlAlchemyStore):
    from mlflow.gateway.guardrails import _SANITIZE_BYPASS_HEADER

    endpoint = _setup_endpoint(store, "ep-bypass")
    mock_response = _make_chat_response("Bypass response")
    mock_request = _make_mock_request(
        {"messages": [{"role": "user", "content": "hello"}]},
        headers={_SANITIZE_BYPASS_HEADER: "1"},
    )

    with patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create:
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )

        with patch("mlflow.server.gateway_api._load_guardrails") as mock_load:
            response = await invocations(endpoint.name, mock_request)
            mock_load.assert_not_called()

    assert response.choices[0].message.content == "Bypass response"


@pytest.mark.asyncio
async def test_invocations_no_guardrails_calls_llm(store: SqlAlchemyStore):
    endpoint = _setup_endpoint(store, "ep-no-guardrails")

    mock_response = _make_chat_response("Direct response")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    with patch("mlflow.server.gateway_api._create_provider_from_endpoint_name") as mock_create:
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create.return_value = (
            mock_provider,
            GatewayEndpointConfig(
                endpoint_id=endpoint.endpoint_id, endpoint_name=endpoint.name, models=[]
            ),
        )

        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "Direct response"
    assert mock_provider.chat.called
