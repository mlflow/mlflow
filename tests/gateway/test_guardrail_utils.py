import json
from unittest import mock

import pytest

from mlflow.entities.gateway_guardrail import (
    GatewayGuardrail,
    GatewayGuardrailConfig,
    GuardrailAction,
    GuardrailStage,
)
from mlflow.entities.scorer import ScorerVersion
from mlflow.gateway.guardrail_utils import (
    load_guardrails,
    run_after_guardrails,
    run_before_guardrails,
)
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail
from mlflow.gateway.schemas.chat import ResponsePayload

# ─── Helpers ─────────────────────────────────────────────────────────────────


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


def _make_scorer_version():
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
        scorer=_make_scorer_version(),
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


# ─── run_before_guardrails ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_before_guardrails_passes():
    g = _make_judge("BEFORE")
    payload = _make_request_payload()
    result = await run_before_guardrails([g], payload)
    assert result == payload
    assert g.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_before_guardrails_skips_after_stage():
    g = _make_judge("AFTER")
    payload = _make_request_payload()
    result = await run_before_guardrails([g], payload)
    assert result == payload
    assert g.scorer.call_count == 0


@pytest.mark.asyncio
async def test_run_before_guardrails_blocks():
    g = _make_judge("BEFORE", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked"):
        await run_before_guardrails([g], _make_request_payload())


@pytest.mark.asyncio
async def test_run_before_guardrails_chains_multiple():
    g1 = _make_judge("BEFORE")
    g2 = _make_judge("BEFORE")
    payload = _make_request_payload()
    result = await run_before_guardrails([g1, g2], payload)
    assert result == payload
    assert g1.scorer.call_count == 1
    assert g2.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_before_guardrails_stops_at_first_failure():
    g1 = _make_judge("BEFORE", passing=False)
    g2 = _make_judge("BEFORE")
    with pytest.raises(GuardrailViolation, match="blocked"):
        await run_before_guardrails([g1, g2], _make_request_payload())
    assert g2.scorer.call_count == 0


# ─── run_after_guardrails ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_after_guardrails_passes():
    g = _make_judge("AFTER")
    req = _make_request_payload()
    response = _make_response_payload()
    result = await run_after_guardrails([g], req, response)
    assert result.choices[0].message.content == "hi there"
    assert g.scorer.call_count == 1


@pytest.mark.asyncio
async def test_run_after_guardrails_skips_before_stage():
    g = _make_judge("BEFORE")
    response = _make_response_payload()
    result = await run_after_guardrails([g], _make_request_payload(), response)
    assert result is response
    assert g.scorer.call_count == 0


@pytest.mark.asyncio
async def test_run_after_guardrails_blocks():
    g = _make_judge("AFTER", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked"):
        await run_after_guardrails([g], _make_request_payload(), _make_response_payload())


@pytest.mark.asyncio
async def test_run_after_guardrails_no_guardrails_returns_response():
    response = _make_response_payload()
    result = await run_after_guardrails([], _make_request_payload(), response)
    assert result is response


# ─── load_guardrails ──────────────────────────────────────────────────────────


def test_load_guardrails_empty():
    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = []
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    result = load_guardrails(store, endpoint_config, request)
    assert result == []
    store.list_endpoint_guardrail_configs.assert_called_once_with("ep-1")


def test_load_guardrails_converts_real_entity():
    config = _make_guardrail_config(stage="BEFORE", action="VALIDATION")
    resolved_scorer = mock.MagicMock()
    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = [config]
    store.resolve_endpoint_in_scorer.return_value = resolved_scorer
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    with mock.patch("mlflow.gateway.guardrails.JudgeGuardrail.from_entity") as mock_from_entity:
        mock_from_entity.return_value = _make_judge("BEFORE")
        result = load_guardrails(store, endpoint_config, request)

    assert len(result) == 1
    assert isinstance(result[0], JudgeGuardrail)
    store.resolve_endpoint_in_scorer.assert_called_once_with(config.guardrail.scorer)
    called_guardrail, called_url = mock_from_entity.call_args[0]
    assert called_url == "http://localhost:5000"
    assert called_guardrail.scorer is resolved_scorer
    assert called_guardrail.guardrail_id == config.guardrail.guardrail_id


def test_load_guardrails_resolves_endpoint_id_to_name():
    """Scorer's gateway:/ model URI contains an endpoint ID; load_guardrails must
    resolve it to the endpoint name via resolve_endpoint_in_scorer before invoking
    from_entity, so the judge self-call uses the correct model name.
    """
    config = _make_guardrail_config(stage="BEFORE", action="VALIDATION")
    original_scorer = config.guardrail.scorer

    resolved_scorer = ScorerVersion(
        experiment_id=original_scorer.experiment_id,
        scorer_name=original_scorer.scorer_name,
        scorer_version=original_scorer.scorer_version,
        serialized_scorer=json.dumps({
            "name": "safety",
            "instructions_judge_pydantic_data": {
                "instructions": "check safety",
                "model": "gateway:/my-endpoint-name",
            },
        }),
        creation_time=original_scorer.creation_time,
        scorer_id=original_scorer.scorer_id,
    )

    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs.return_value = [config]
    store.resolve_endpoint_in_scorer.return_value = resolved_scorer
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    with mock.patch("mlflow.gateway.guardrails.JudgeGuardrail.from_entity") as mock_from_entity:
        mock_from_entity.return_value = _make_judge("BEFORE")
        load_guardrails(store, endpoint_config, request)

    store.resolve_endpoint_in_scorer.assert_called_once_with(original_scorer)
    called_guardrail, _ = mock_from_entity.call_args[0]
    assert called_guardrail.scorer is resolved_scorer


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
        if entity.stage == GuardrailStage.AFTER:
            raise ValueError("bad scorer")
        return good_judge

    with mock.patch(
        "mlflow.gateway.guardrails.JudgeGuardrail.from_entity",
        side_effect=from_entity_side_effect,
    ):
        result = load_guardrails(store, endpoint_config, request)

    assert result == [good_judge]
