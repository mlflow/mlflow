import json
import uuid
from typing import Any
from unittest import mock

import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment import Feedback
from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail
from mlflow.tracing.client import TracingClient
from mlflow.types.chat import ChatCompletionResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(text="Hello, world!"):
    return {"messages": [{"role": "user", "content": text}]}


def _make_response(text="I'm a helpful assistant."):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }


class _SimpleScorer:
    """Minimal scorer that returns a fixed value and tracks call count."""

    def __init__(self, return_value: Any) -> None:
        self.call_count = 0
        self._return_value = return_value

    def __call__(self, **kwargs) -> Any:
        self.call_count += 1
        return self._return_value


def _feedback(value, rationale="some rationale"):
    return Feedback(value=value, rationale=rationale)


# ---------------------------------------------------------------------------
# BEFORE / VALIDATION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_before_validation_pass():
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    req = _make_request()
    result = await guard.process_request(req)
    assert result is req
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_before_validation_block():
    scorer = _SimpleScorer(_feedback(value=False, rationale="toxic content"))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, name="safety")
    with pytest.raises(GuardrailViolation, match="safety.*toxic content"):
        await guard.process_request(_make_request())
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_before_validation_skips_response():
    scorer = _SimpleScorer(_feedback(value=False))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    resp = _make_response()
    result = await guard.process_response(_make_request(), resp)
    assert result is resp
    assert scorer.call_count == 0


# ---------------------------------------------------------------------------
# AFTER / VALIDATION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_validation_pass():
    scorer = _SimpleScorer(_feedback(value="yes"))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, "test")
    req = _make_request("What is 2+2?")
    resp = _make_response("4")
    result = await guard.process_response(req, resp)
    assert result is resp
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_after_validation_block():
    scorer = _SimpleScorer(_feedback(value="no", rationale="PII detected"))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, name="pii")
    with pytest.raises(GuardrailViolation, match="pii.*PII detected"):
        await guard.process_response(_make_request(), _make_response())
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_after_validation_skips_request():
    scorer = _SimpleScorer(_feedback(value=False))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, "test")
    req = _make_request()
    result = await guard.process_request(req)
    assert result is req
    assert scorer.call_count == 0


# ---------------------------------------------------------------------------
# SANITIZATION
# ---------------------------------------------------------------------------


def _send_request_returning(payload):
    return mock.AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(payload)}}]})


@pytest.mark.asyncio
async def test_before_sanitization_rewrites_request():
    scorer = _SimpleScorer(_feedback(value=False, rationale="contains PII"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    sanitized = _make_request("my SSN is [REDACTED]")
    with mock.patch("mlflow.gateway.guardrails.send_request", _send_request_returning(sanitized)):
        result = await guard.process_request(_make_request("my SSN is 123-45-6789"))
    assert result == sanitized


@pytest.mark.asyncio
async def test_after_sanitization_rewrites_response():
    scorer = _SimpleScorer(_feedback(value=False, rationale="toxic language"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.AFTER,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    sanitized = _make_response("Polite version")
    with mock.patch("mlflow.gateway.guardrails.send_request", _send_request_returning(sanitized)):
        result = await guard.process_response(_make_request(), _make_response("rude text"))
    assert result == sanitized


@pytest.mark.asyncio
async def test_sanitization_without_endpoint_raises():
    scorer = _SimpleScorer(_feedback(value=False, rationale="issue found"))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION, "test")
    with pytest.raises(GuardrailViolation, match="action_llm_url"):
        await guard.process_request(_make_request())


@pytest.mark.asyncio
async def test_sanitization_invalid_json_raises():
    scorer = _SimpleScorer(_feedback(value=False, rationale="fix"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    with (
        mock.patch(
            "mlflow.gateway.guardrails.send_request",
            mock.AsyncMock(return_value={"choices": [{"message": {"content": "not json"}}]}),
        ),
        pytest.raises(GuardrailViolation, match="invalid JSON"),
    ):
        await guard.process_request(_make_request())


@pytest.mark.asyncio
async def test_sanitization_network_error_raises():
    from fastapi import HTTPException

    scorer = _SimpleScorer(_feedback(value=False, rationale="issue"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    with (
        mock.patch(
            "mlflow.gateway.guardrails.send_request",
            side_effect=HTTPException(status_code=503, detail="timed out"),
        ),
        pytest.raises(GuardrailViolation, match="Sanitization request failed"),
    ):
        await guard.process_request(_make_request())


@pytest.mark.asyncio
async def test_sanitization_passes_on_good_content():
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION, "test")
    req = _make_request()
    assert await guard.process_request(req) is req


@pytest.mark.asyncio
async def test_sanitization_skips_response_format_when_no_schema_provided():
    # When payload_schema is None (the default), sanitization omits response_format.
    # Used by passthrough endpoints, where ChatCompletionRequest shares field names
    # with provider-specific shapes (e.g. Anthropic also uses messages/max_tokens),
    # making reliable detection impossible — so callers explicitly opt in via payload_schema.
    scorer = _SimpleScorer(_feedback(value=False, rationale="issue"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    sanitized = _make_request("cleaned")
    captured: list[dict[str, Any]] = []

    async def capture_send_request(*args, **kwargs):
        captured.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(sanitized)}}]}

    with mock.patch("mlflow.gateway.guardrails.send_request", side_effect=capture_send_request):
        await guard.process_request(_make_request())

    assert "response_format" not in captured[0]["payload"]


def _make_full_response(text="I'm a helpful assistant."):
    """Return a response dict that satisfies ChatCompletionResponse validation."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


@pytest.mark.asyncio
async def test_sanitization_uses_response_format_for_chat_response():
    scorer = _SimpleScorer(_feedback(value=False, rationale="issue"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.AFTER,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    sanitized = _make_full_response("cleaned")
    captured: list[dict[str, Any]] = []

    async def capture_send_request(*args, **kwargs):
        captured.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(sanitized)}}]}

    with mock.patch("mlflow.gateway.guardrails.send_request", side_effect=capture_send_request):
        await guard.process_response(
            _make_request(),
            _make_full_response("bad"),
            payload_schema=ChatCompletionResponse.model_json_schema(),
        )

    assert captured[0]["payload"]["response_format"]["json_schema"]["schema"] == (
        ChatCompletionResponse.model_json_schema()
    )


@pytest.mark.asyncio
async def test_sanitization_skips_response_format_for_passthrough_payload():
    scorer = _SimpleScorer(_feedback(value=False, rationale="issue"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "test",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    # Anthropic-style payload that doesn't conform to ChatCompletionRequest
    anthropic_request = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
    }
    sanitized = {**anthropic_request}
    captured: list[dict[str, Any]] = []

    async def capture_send_request(*args, **kwargs):
        captured.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(sanitized)}}]}

    with mock.patch("mlflow.gateway.guardrails.send_request", side_effect=capture_send_request):
        await guard.process_request(anthropic_request)

    assert "response_format" not in captured[0]["payload"]


# ---------------------------------------------------------------------------
# _is_passing with Feedback values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected_pass"),
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("no", False),
        ("unknown", False),
        ("anything_else", False),
    ],
)
async def test_is_passing_feedback_values(value, expected_pass):
    scorer = _SimpleScorer(_feedback(value=value))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    if expected_pass:
        result = await guard.process_request(_make_request())
        assert result is not None
    else:
        with pytest.raises(GuardrailViolation, match="blocked"):
            await guard.process_request(_make_request())
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_unexpected_feedback_value_type_raises():
    scorer = _SimpleScorer(_feedback(value=1))  # int inside Feedback is not supported
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    with pytest.raises(TypeError, match="unexpected value type"):
        await guard.process_request(_make_request())


# ---------------------------------------------------------------------------
# Plain scalar return values (scorer returns bool/str directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected_pass"),
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("no", False),
    ],
)
async def test_is_passing_plain_scalar(value, expected_pass):
    scorer = _SimpleScorer(value)
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    if expected_pass:
        result = await guard.process_request(_make_request())
        assert result is not None
    else:
        with pytest.raises(GuardrailViolation, match="blocked"):
            await guard.process_request(_make_request())
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_unexpected_scorer_type_raises():
    scorer = _SimpleScorer(42)  # int is not a supported return type
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    with pytest.raises(TypeError, match="unexpected value type"):
        await guard.process_request(_make_request())


# ---------------------------------------------------------------------------
# list[Feedback] return value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_feedback_all_pass():
    scorer = _SimpleScorer([_feedback(value=True), _feedback(value="yes")])
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    result = await guard.process_request(_make_request())
    assert result is not None
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_list_feedback_one_fails():
    scorer = _SimpleScorer([
        _feedback(value=True),
        _feedback(value=False, rationale="unsafe"),
    ])
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, name="multi")
    with pytest.raises(GuardrailViolation, match="multi.*unsafe"):
        await guard.process_request(_make_request())
    assert scorer.call_count == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_messages_request():
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "test")
    result = await guard.process_request({"messages": []})
    assert result == {"messages": []}
    assert scorer.call_count == 1


@pytest.mark.asyncio
async def test_empty_choices_response():
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, "test")
    result = await guard.process_response(_make_request(), {"choices": []})
    assert result == {"choices": []}
    assert scorer.call_count == 1


# ---------------------------------------------------------------------------
# from_entity conversion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_entity():
    mock_serialized_scorer = mock.MagicMock()
    mock_scorer_version = mock.MagicMock()
    mock_scorer_version.serialized_scorer = mock_serialized_scorer

    entity = mock.MagicMock()
    entity.scorer = mock_scorer_version
    entity.name = "safety-guard"
    entity.stage = GuardrailStage.BEFORE
    entity.action = GuardrailAction.VALIDATION
    entity.action_endpoint_name = None

    with mock.patch(
        "mlflow.genai.scorers.Scorer.model_validate",
        return_value=_SimpleScorer(_feedback(value=True)),
    ) as mock_validate:
        guard = JudgeGuardrail.from_entity(entity)
        mock_validate.assert_called_once_with(mock_serialized_scorer)

    assert isinstance(guard, JudgeGuardrail)
    assert guard.stage == GuardrailStage.BEFORE
    assert guard.action == GuardrailAction.VALIDATION
    assert guard.name == "safety-guard"
    assert guard.action_llm_url is None

    result = await guard.process_request(_make_request())
    assert result is not None


def test_from_entity_with_action_endpoint():
    mock_serialized_scorer = mock.MagicMock()
    mock_scorer_version = mock.MagicMock()
    mock_scorer_version.serialized_scorer = mock_serialized_scorer

    entity = mock.MagicMock()
    entity.scorer = mock_scorer_version
    entity.name = "sanitizer-guard"
    entity.stage = GuardrailStage.BEFORE
    entity.action = GuardrailAction.SANITIZATION
    entity.action_endpoint_name = "my-ep"

    with mock.patch(
        "mlflow.genai.scorers.Scorer.model_validate",
        return_value=_SimpleScorer(_feedback(value=True)),
    ):
        guard = JudgeGuardrail.from_entity(entity, server_url="http://localhost:5000")

    assert guard.action_llm_url == "http://localhost:5000"
    assert guard.action_endpoint_name == "my-ep"


def test_from_entity_rewrites_gateway_model_uri():
    """gateway:/ model URIs are kept as gateway:/ but given an explicit base_url so
    _get_provider_instance can skip _resolve_gateway_uri(), which fails when
    MLFLOW_TRACKING_URI is the backend store URI (e.g. sqlite://) inside the server process.
    """
    from mlflow.genai.judges.instructions_judge import InstructionsJudge

    mock_instructions_judge = mock.MagicMock(spec=InstructionsJudge)
    mock_instructions_judge.model = "gateway:/my-judge-ep"
    mock_instructions_judge.name = "my-judge"
    mock_instructions_judge._instructions = "Is this safe? {{ inputs }}"
    mock_instructions_judge._feedback_value_type = None
    mock_instructions_judge._inference_params = None

    entity = mock.MagicMock()
    entity.scorer.serialized_scorer = {}
    entity.name = "safety-guard"
    entity.stage = GuardrailStage.BEFORE
    entity.action = GuardrailAction.VALIDATION
    entity.action_endpoint_name = None

    with mock.patch(
        "mlflow.genai.scorers.Scorer.model_validate", return_value=mock_instructions_judge
    ):
        guard = JudgeGuardrail.from_entity(entity, server_url="http://localhost:5000")

    assert isinstance(guard.scorer, InstructionsJudge)
    assert guard.scorer.model == "gateway:/my-judge-ep"
    assert guard.scorer._base_url == "http://localhost:5000/gateway/mlflow/v1/chat/completions"


def test_from_entity_does_not_rewrite_non_gateway_model_uri():
    from mlflow.genai.judges.instructions_judge import InstructionsJudge

    mock_instructions_judge = mock.MagicMock(spec=InstructionsJudge)
    mock_instructions_judge.model = "openai:/gpt-4o"
    mock_instructions_judge.name = "my-judge"
    mock_instructions_judge._instructions = "Is this safe? {{ inputs }}"
    mock_instructions_judge._feedback_value_type = None
    mock_instructions_judge._inference_params = None

    entity = mock.MagicMock()
    entity.scorer.serialized_scorer = {}
    entity.name = "safety-guard"
    entity.stage = GuardrailStage.BEFORE
    entity.action = GuardrailAction.VALIDATION
    entity.action_endpoint_name = None

    with mock.patch(
        "mlflow.genai.scorers.Scorer.model_validate", return_value=mock_instructions_judge
    ):
        guard = JudgeGuardrail.from_entity(entity, server_url="http://localhost:5000")

    assert guard.scorer is mock_instructions_judge


# ---------------------------------------------------------------------------
# Tracing: spans created during guardrail execution
# ---------------------------------------------------------------------------


@pytest.fixture
def tracing_experiment():
    exp_id = mlflow.create_experiment(f"guardrail-tracing-{uuid.uuid4()}")
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def _get_span_map(experiment_id):
    traces = TracingClient().search_traces(locations=[experiment_id])
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"
    return {s.name: s for s in traces[0].data.spans}


@pytest.mark.asyncio
async def test_process_request_creates_guardrail_and_judge_spans(tracing_experiment):
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "safety")

    @mlflow.trace
    async def _run():
        return await guard.process_request(_make_request(), usage_tracking=True)

    result = await _run()
    assert result == _make_request()

    spans = _get_span_map(tracing_experiment)
    assert "guardrail/safety" in spans
    assert "judge" in spans

    gspan = spans["guardrail/safety"]
    jspan = spans["judge"]
    assert gspan.span_type == SpanType.GUARDRAIL
    assert jspan.span_type == SpanType.EVALUATOR
    assert jspan.outputs == {"passed": True, "rationale": "some rationale"}
    assert jspan.parent_id == gspan.span_id


@pytest.mark.asyncio
async def test_process_request_no_spans_when_usage_tracking_off(tracing_experiment):
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, "safety")
    result = await guard.process_request(_make_request(), usage_tracking=False)
    assert result == _make_request()

    traces = TracingClient().search_traces(locations=[tracing_experiment])
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_process_response_creates_guardrail_and_judge_spans(tracing_experiment):
    scorer = _SimpleScorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, "pii")

    @mlflow.trace
    async def _run():
        return await guard.process_response(_make_request(), _make_response(), usage_tracking=True)

    await _run()

    spans = _get_span_map(tracing_experiment)
    assert "guardrail/pii" in spans
    assert "judge" in spans

    gspan = spans["guardrail/pii"]
    jspan = spans["judge"]
    assert gspan.span_type == SpanType.GUARDRAIL
    assert jspan.span_type == SpanType.EVALUATOR
    assert jspan.parent_id == gspan.span_id


@pytest.mark.asyncio
async def test_sanitization_creates_span_when_usage_tracking_on(tracing_experiment):
    scorer = _SimpleScorer(_feedback(value=False, rationale="contains PII"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        "pii-guard",
        action_llm_url="http://localhost:5000",
        action_endpoint_name="ep-sanitizer",
    )
    sanitized = _make_request("my SSN is [REDACTED]")

    @mlflow.trace
    async def _run():
        with mock.patch(
            "mlflow.gateway.guardrails.send_request", _send_request_returning(sanitized)
        ):
            return await guard.process_request(
                _make_request("my SSN is 123-45-6789"), usage_tracking=True
            )

    result = await _run()
    assert result == sanitized

    spans = _get_span_map(tracing_experiment)
    assert "guardrail/pii-guard" in spans
    assert "judge" in spans
    assert "sanitization" in spans

    gspan = spans["guardrail/pii-guard"]
    jspan = spans["judge"]
    san_span = spans["sanitization"]
    assert san_span.span_type == SpanType.LLM
    assert jspan.outputs == {"passed": False, "rationale": "contains PII"}
    assert jspan.parent_id == gspan.span_id
    assert san_span.parent_id == gspan.span_id
