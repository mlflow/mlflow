from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail, from_entity

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


def _mock_scorer(return_value):
    scorer = mock.MagicMock()
    scorer.return_value = return_value
    return scorer


def _feedback(value, rationale="some rationale"):
    return Feedback(value=value, rationale=rationale)


# ---------------------------------------------------------------------------
# BEFORE / VALIDATION
# ---------------------------------------------------------------------------


def test_before_validation_pass():
    scorer = _mock_scorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    req = _make_request()
    result = guard.process_request(req)
    assert result is req
    scorer.assert_called_once_with(outputs="Hello, world!")


def test_before_validation_block():
    scorer = _mock_scorer(_feedback(value=False, rationale="toxic content"))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, name="safety")
    with pytest.raises(GuardrailViolation, match="safety.*toxic content"):
        guard.process_request(_make_request())
    scorer.assert_called_once()


def test_before_validation_skips_response():
    scorer = _mock_scorer(_feedback(value=False))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    resp = _make_response()
    result = guard.process_response(resp)
    assert result is resp
    scorer.assert_not_called()


# ---------------------------------------------------------------------------
# AFTER / VALIDATION
# ---------------------------------------------------------------------------


def test_after_validation_pass():
    scorer = _mock_scorer(_feedback(value="yes"))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
    resp = _make_response()
    result = guard.process_response(resp)
    assert result is resp
    scorer.assert_called_once_with(outputs="I'm a helpful assistant.")


def test_after_validation_block():
    scorer = _mock_scorer(_feedback(value="no", rationale="PII detected"))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, name="pii")
    with pytest.raises(GuardrailViolation, match="pii.*PII detected"):
        guard.process_response(_make_response())
    scorer.assert_called_once()


def test_after_validation_skips_request():
    scorer = _mock_scorer(_feedback(value=False))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
    req = _make_request()
    result = guard.process_request(req)
    assert result is req
    scorer.assert_not_called()


# ---------------------------------------------------------------------------
# SANITIZATION (currently raises)
# ---------------------------------------------------------------------------


def _mock_endpoint_config():
    """Create a mock endpoint config with provider/model info."""
    model_config = mock.MagicMock()
    model_config.provider = "openai"
    model_config.model_name = "gpt-4"
    config = mock.MagicMock()
    config.model_configs = [model_config]
    return config


def _patch_sanitization(rewritten_text="Cleaned text"):
    """Context manager that mocks both get_endpoint_config and _invoke_via_gateway."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            mock.patch(
                "mlflow.store.tracking.gateway.config_resolver.get_endpoint_config",
                return_value=_mock_endpoint_config(),
            ),
            mock.patch(
                "mlflow.genai.judges.adapters.gateway_adapter._invoke_via_gateway",
                return_value=rewritten_text,
            ),
        ):
            yield

    return _ctx()


def test_before_sanitization_rewrites_request():
    scorer = _mock_scorer(_feedback(value=False, rationale="contains PII"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        action_endpoint_id="ep-sanitizer",
    )
    with _patch_sanitization("Cleaned text"):
        result = guard.process_request(_make_request("my SSN is 123-45-6789"))
    assert result["messages"][-1]["content"] == "Cleaned text"


def test_after_sanitization_rewrites_response():
    scorer = _mock_scorer(_feedback(value=False, rationale="toxic language"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.AFTER,
        GuardrailAction.SANITIZATION,
        action_endpoint_id="ep-sanitizer",
    )
    with _patch_sanitization("Polite version"):
        result = guard.process_response(_make_response("rude text"))
    assert result["choices"][0]["message"]["content"] == "Polite version"


def test_sanitization_without_endpoint_raises():
    scorer = _mock_scorer(_feedback(value=False, rationale="issue found"))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION)
    with pytest.raises(GuardrailViolation, match="action_endpoint_id"):
        guard.process_request(_make_request())


def test_sanitization_does_not_mutate_original():
    scorer = _mock_scorer(_feedback(value=False, rationale="fix"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        action_endpoint_id="ep-sanitizer",
    )
    original = _make_request("original text")
    with _patch_sanitization("rewritten"):
        result = guard.process_request(original)
    assert original["messages"][-1]["content"] == "original text"
    assert result["messages"][-1]["content"] == "rewritten"


def test_sanitization_calls_invoke_via_gateway_with_correct_model_uri():
    scorer = _mock_scorer(_feedback(value=False, rationale="needs fix"))
    guard = JudgeGuardrail(
        scorer,
        GuardrailStage.BEFORE,
        GuardrailAction.SANITIZATION,
        action_endpoint_id="ep-sanitizer",
    )
    with (
        mock.patch(
            "mlflow.store.tracking.gateway.config_resolver.get_endpoint_config",
            return_value=_mock_endpoint_config(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._invoke_via_gateway",
            return_value="fixed text",
        ) as mock_invoke,
    ):
        guard.process_request(_make_request("bad text"))
    mock_invoke.assert_called_once()
    call_args = mock_invoke.call_args
    assert call_args[0][0] == "openai:/gpt-4"
    assert call_args[0][1] == "openai"


def test_sanitization_passes_on_good_content():
    scorer = _mock_scorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION)
    req = _make_request()
    assert guard.process_request(req) is req


# ---------------------------------------------------------------------------
# _is_passing with Feedback values
# ---------------------------------------------------------------------------


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
        (1, True),
        (0, False),
    ],
)
def test_is_passing_feedback_values(value, expected_pass):
    scorer = _mock_scorer(_feedback(value=value))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    if expected_pass:
        result = guard.process_request(_make_request())
        assert result is not None
    else:
        with pytest.raises(GuardrailViolation, match="blocked"):
            guard.process_request(_make_request())


# ---------------------------------------------------------------------------
# Plain scalar return values (scorer returns int/bool/str directly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_pass"),
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
    ],
)
def test_is_passing_plain_scalar(value, expected_pass):
    scorer = _mock_scorer(value)
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    if expected_pass:
        result = guard.process_request(_make_request())
        assert result is not None
    else:
        with pytest.raises(GuardrailViolation, match="blocked"):
            guard.process_request(_make_request())


# ---------------------------------------------------------------------------
# list[Feedback] return value
# ---------------------------------------------------------------------------


def test_list_feedback_all_pass():
    scorer = _mock_scorer([_feedback(value=True), _feedback(value="yes")])
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    result = guard.process_request(_make_request())
    assert result is not None


def test_list_feedback_one_fails():
    scorer = _mock_scorer([
        _feedback(value=True),
        _feedback(value=False, rationale="unsafe"),
    ])
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, name="multi")
    with pytest.raises(GuardrailViolation, match="multi.*unsafe"):
        guard.process_request(_make_request())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_messages_request():
    scorer = _mock_scorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    result = guard.process_request({"messages": []})
    assert result == {"messages": []}
    scorer.assert_called_once_with(outputs="")


def test_empty_choices_response():
    scorer = _mock_scorer(_feedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
    result = guard.process_response({"choices": []})
    assert result == {"choices": []}
    scorer.assert_called_once_with(outputs="")


# ---------------------------------------------------------------------------
# from_entity conversion
# ---------------------------------------------------------------------------


def test_from_entity():
    mock_serialized_scorer = mock.MagicMock()
    mock_scorer_version = mock.MagicMock()
    mock_scorer_version.serialized_scorer = mock_serialized_scorer

    entity = mock.MagicMock()
    entity.scorer = mock_scorer_version
    entity.name = "safety-guard"
    entity.stage = GuardrailStage.BEFORE
    entity.action = GuardrailAction.VALIDATION
    entity.guardrail_id = "gr-abc123"
    entity.action_endpoint_id = None

    with mock.patch(
        "mlflow.genai.scorers.Scorer.model_validate",
        return_value=_mock_scorer(_feedback(value=True)),
    ) as mock_validate:
        guard = from_entity(entity)
        mock_validate.assert_called_once_with(mock_serialized_scorer)

    assert isinstance(guard, JudgeGuardrail)
    assert guard.stage == GuardrailStage.BEFORE
    assert guard.action == GuardrailAction.VALIDATION
    assert guard.name == "safety-guard"

    result = guard.process_request(_make_request())
    assert result is not None
