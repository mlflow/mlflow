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


# ─── Real-DB end-to-end scenarios ─────────────────────────────────────────────
#
# These tests use a real SqlAlchemyStore (no _load_guardrails or
# _create_provider_from_endpoint_name mocks).  Only the two external I/O
# boundaries are patched:
#   - Scorer.model_validate  → returns a _SimpleScorer so we don't need a
#                              live judge LLM to deserialize the stored scorer.
#   - OpenAIProvider.chat    → returns a canned response so we don't need
#                              a real OpenAI key.
# ─────────────────────────────────────────────────────────────────────────────


_SERIALIZED_SCORER = json.dumps({"name": "safety", "builtin_scorer_class": "Safety"})


def _setup_db_guardrail(
    store: SqlAlchemyStore,
    endpoint_name: str,
    stage: str,
    action: str,
    action_endpoint_name: str | None = None,
):
    """Create scorer + guardrail in DB and attach it to the endpoint."""
    experiment_id = store.create_experiment(f"exp-{endpoint_name}-{stage}")
    scorer_ver = store.register_scorer(experiment_id, f"scorer-{endpoint_name}", _SERIALIZED_SCORER)

    action_endpoint_id = None
    if action_endpoint_name:
        sanitizer_ep = store.get_gateway_endpoint(name=action_endpoint_name)
        action_endpoint_id = sanitizer_ep.endpoint_id

    guardrail = store.create_gateway_guardrail(
        name=f"guardrail-{endpoint_name}-{stage}",
        scorer_id=scorer_ver.scorer_id,
        scorer_version=scorer_ver.scorer_version,
        stage=GuardrailStage(stage),
        action=GuardrailAction(action),
        action_endpoint_id=action_endpoint_id,
    )
    endpoint = store.get_gateway_endpoint(name=endpoint_name)
    store.add_guardrail_to_endpoint(endpoint.endpoint_id, guardrail.guardrail_id)
    return guardrail, scorer_ver


@pytest.mark.asyncio
async def test_real_db_before_guardrail_passes(store: SqlAlchemyStore):
    endpoint = _setup_endpoint(store, "real-ep-before-pass")
    _setup_db_guardrail(store, "real-ep-before-pass", "BEFORE", "VALIDATION")

    mock_response = _make_chat_response("Safe response")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    passing_scorer = _SimpleScorer(passing=True)

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=passing_scorer),
        patch(
            "mlflow.gateway.providers.openai.OpenAIProvider.chat",
            AsyncMock(return_value=mock_response),
        ),
    ):
        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "Safe response"
    assert passing_scorer.call_count == 1


@pytest.mark.asyncio
async def test_real_db_before_guardrail_blocks(store: SqlAlchemyStore):
    from fastapi import HTTPException

    endpoint = _setup_endpoint(store, "real-ep-before-block")
    _setup_db_guardrail(store, "real-ep-before-block", "BEFORE", "VALIDATION")

    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "bad input"}]})

    blocking_scorer = _SimpleScorer(passing=False)

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=blocking_scorer),
        patch(
            "mlflow.gateway.providers.openai.OpenAIProvider.chat",
            AsyncMock(),
        ) as mock_chat,
    ):
        with pytest.raises(HTTPException, match="400"):
            await invocations(endpoint.name, mock_request)

    assert not mock_chat.called
    assert blocking_scorer.call_count == 1


@pytest.mark.asyncio
async def test_real_db_after_guardrail_blocks(store: SqlAlchemyStore):
    from fastapi import HTTPException

    endpoint = _setup_endpoint(store, "real-ep-after-block")
    _setup_db_guardrail(store, "real-ep-after-block", "AFTER", "VALIDATION")

    mock_response = _make_chat_response("Unsafe output")
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    blocking_scorer = _SimpleScorer(passing=False)

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=blocking_scorer),
        patch(
            "mlflow.gateway.providers.openai.OpenAIProvider.chat",
            AsyncMock(return_value=mock_response),
        ) as mock_chat,
    ):
        with pytest.raises(HTTPException, match="400"):
            await invocations(endpoint.name, mock_request)

    assert mock_chat.called
    assert blocking_scorer.call_count == 1


# ─── Sanitization end-to-end scenarios ────────────────────────────────────────


@pytest.mark.asyncio
async def test_invocations_before_sanitize_rewrites_request(store: SqlAlchemyStore):
    """BEFORE/SANITIZATION: failing scorer triggers _sanitize which rewrites the request."""
    endpoint = _setup_endpoint(store, "ep-sanitize-before")
    sanitizer = _setup_endpoint(store, "ep-sanitizer")
    _setup_db_guardrail(
        store, "ep-sanitize-before", "BEFORE", "SANITIZATION", action_endpoint_name=sanitizer.name
    )

    original_body = {"messages": [{"role": "user", "content": "bad input"}]}
    sanitized_body = {"messages": [{"role": "user", "content": "cleaned input"}]}
    mock_response = _make_chat_response("Response to cleaned input")
    mock_request = _make_mock_request(original_body)

    failing_scorer = _SimpleScorer(passing=False)
    captured_chat_payloads: list = []

    async def fake_chat(payload):
        captured_chat_payloads.append(payload)
        return mock_response

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=failing_scorer),
        patch(
            "mlflow.gateway.guardrails.send_request",
            AsyncMock(
                return_value={"choices": [{"message": {"content": json.dumps(sanitized_body)}}]}
            ),
        ),
        patch("mlflow.gateway.providers.openai.OpenAIProvider.chat", side_effect=fake_chat),
    ):
        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "Response to cleaned input"
    assert failing_scorer.call_count == 1
    # The provider received the sanitized (rewritten) payload, not the original
    assert captured_chat_payloads[0].messages[0].content == "cleaned input"


@pytest.mark.asyncio
async def test_invocations_after_sanitize_rewrites_response(store: SqlAlchemyStore):
    """AFTER/SANITIZATION: failing scorer triggers _sanitize which rewrites the response."""
    endpoint = _setup_endpoint(store, "ep-sanitize-after")
    sanitizer = _setup_endpoint(store, "ep-sanitizer-after")
    _setup_db_guardrail(
        store, "ep-sanitize-after", "AFTER", "SANITIZATION", action_endpoint_name=sanitizer.name
    )

    original_content = "rude output"
    sanitized_response = {
        "id": "resp-sanitized",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "polite output"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    mock_response = _make_chat_response(original_content)
    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "hello"}]})

    failing_scorer = _SimpleScorer(passing=False)

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=failing_scorer),
        patch(
            "mlflow.gateway.guardrails.send_request",
            AsyncMock(
                return_value={
                    "choices": [{"message": {"content": json.dumps(sanitized_response)}}]
                }
            ),
        ),
        patch(
            "mlflow.gateway.providers.openai.OpenAIProvider.chat",
            AsyncMock(return_value=mock_response),
        ),
    ):
        response = await invocations(endpoint.name, mock_request)

    assert response.choices[0].message.content == "polite output"
    assert failing_scorer.call_count == 1


@pytest.mark.asyncio
async def test_invocations_sanitize_no_action_endpoint_blocks(store: SqlAlchemyStore):
    """SANITIZATION without action_endpoint_name raises 400."""
    from fastapi import HTTPException

    endpoint = _setup_endpoint(store, "ep-sanitize-no-ep")
    _setup_db_guardrail(store, "ep-sanitize-no-ep", "BEFORE", "SANITIZATION")

    mock_request = _make_mock_request({"messages": [{"role": "user", "content": "bad input"}]})

    failing_scorer = _SimpleScorer(passing=False)

    with (
        patch("mlflow.genai.scorers.base.Scorer.model_validate", return_value=failing_scorer),
        patch("mlflow.gateway.providers.openai.OpenAIProvider.chat", AsyncMock()),
    ):
        with pytest.raises(HTTPException, match="400"):
            await invocations(endpoint.name, mock_request)
