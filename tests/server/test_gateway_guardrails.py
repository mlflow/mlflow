from unittest import mock

import pytest

from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.gateway.guardrails import GuardrailViolation
from mlflow.server.gateway_api import (
    _load_guardrails,
    _run_after_guardrails,
    _run_before_guardrails,
)


def _make_request_payload():
    return {"messages": [{"role": "user", "content": "hello"}]}


def _make_response_payload():
    from mlflow.gateway.schemas.chat import ResponsePayload

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


def _mock_guardrail(stage, action="VALIDATION", *, passing=True):
    g = mock.MagicMock()
    g.stage = GuardrailStage(stage)
    g.action = GuardrailAction(action)
    g.name = f"test-{stage.lower()}"

    def process_request(payload, auth_headers=None):
        if not passing:
            raise GuardrailViolation(g.name, "blocked by test")
        return payload

    def process_response(request, response, auth_headers=None):
        if not passing:
            raise GuardrailViolation(g.name, "blocked by test")
        return response

    g.process_request = mock.MagicMock(side_effect=process_request)
    g.process_response = mock.MagicMock(side_effect=process_response)
    return g


def test_run_before_guardrails_passes():
    g = _mock_guardrail("BEFORE")
    payload = _make_request_payload()
    result = _run_before_guardrails([g], payload)
    assert result == payload
    g.process_request.assert_called_once_with(payload, auth_headers=None)


def test_run_before_guardrails_skips_after_stage():
    g = _mock_guardrail("AFTER")
    payload = _make_request_payload()
    result = _run_before_guardrails([g], payload)
    assert result == payload
    g.process_request.assert_not_called()


def test_run_before_guardrails_blocks():
    g = _mock_guardrail("BEFORE", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked by test"):
        _run_before_guardrails([g], _make_request_payload())


def test_run_before_guardrails_chains_multiple():
    g1 = _mock_guardrail("BEFORE")
    g1.process_request = mock.MagicMock(
        side_effect=lambda p, auth_headers=None: {**p, "extra": "from-g1"}
    )
    g2 = _mock_guardrail("BEFORE")
    g2.process_request = mock.MagicMock(
        side_effect=lambda p, auth_headers=None: {**p, "extra2": "from-g2"}
    )

    result = _run_before_guardrails([g1, g2], _make_request_payload())
    assert result["extra"] == "from-g1"
    assert result["extra2"] == "from-g2"


def test_run_after_guardrails_passes():
    g = _mock_guardrail("AFTER")
    req = _make_request_payload()
    response = _make_response_payload()
    result = _run_after_guardrails([g], req, response)
    assert result.choices[0].message.content == "hi there"
    g.process_response.assert_called_once()
    args, _ = g.process_response.call_args
    assert args[0] == req


def test_run_after_guardrails_skips_before_stage():
    g = _mock_guardrail("BEFORE")
    response = _make_response_payload()
    result = _run_after_guardrails([g], _make_request_payload(), response)
    assert result is response
    g.process_response.assert_not_called()


def test_run_after_guardrails_blocks():
    g = _mock_guardrail("AFTER", passing=False)
    with pytest.raises(GuardrailViolation, match="blocked by test"):
        _run_after_guardrails([g], _make_request_payload(), _make_response_payload())


def test_load_guardrails_empty():
    store = mock.MagicMock()
    store.list_endpoint_guardrail_configs = mock.MagicMock(return_value=[])
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    result = _load_guardrails(store, endpoint_config, request)
    assert result == []
    store.list_endpoint_guardrail_configs.assert_called_once_with("ep-1")


def test_load_guardrails_converts_entities():
    store = mock.MagicMock()
    config = mock.MagicMock()
    config.guardrail = mock.MagicMock()
    config.guardrail_id = "gr-1"
    store.list_endpoint_guardrail_configs = mock.MagicMock(return_value=[config])
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    mock_judge = mock.MagicMock()
    with mock.patch(
        "mlflow.server.gateway_api.JudgeGuardrail.from_entity",
        return_value=mock_judge,
    ) as mock_from_entity:
        result = _load_guardrails(store, endpoint_config, request)
        assert result == [mock_judge]
        mock_from_entity.assert_called_once_with(config.guardrail, "http://localhost:5000")


def test_load_guardrails_skips_failed_conversion():
    store = mock.MagicMock()
    good_config = mock.MagicMock()
    good_config.guardrail = mock.MagicMock()
    good_config.guardrail_id = "gr-good"
    bad_config = mock.MagicMock()
    bad_config.guardrail = mock.MagicMock()
    bad_config.guardrail_id = "gr-bad"
    store.list_endpoint_guardrail_configs = mock.MagicMock(return_value=[bad_config, good_config])
    endpoint_config = mock.MagicMock()
    endpoint_config.endpoint_id = "ep-1"
    request = mock.MagicMock()
    request.base_url = "http://localhost:5000/"

    mock_judge = mock.MagicMock()

    def from_entity_side_effect(entity, server_url):
        if entity is bad_config.guardrail:
            raise ValueError("bad scorer")
        return mock_judge

    with mock.patch(
        "mlflow.server.gateway_api.JudgeGuardrail.from_entity",
        side_effect=from_entity_side_effect,
    ):
        result = _load_guardrails(store, endpoint_config, request)
        assert result == [mock_judge]
