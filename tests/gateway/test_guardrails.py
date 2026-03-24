from unittest import mock

import pytest

from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_request(text="Hello, world!"):
    return {"messages": [{"role": "user", "content": text}]}


def _make_response(text="I'm a helpful assistant."):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }


def _mock_scorer(return_value):
    """Create a mock scorer that returns *return_value* when called."""
    scorer = mock.MagicMock()
    scorer.return_value = return_value
    return scorer


class _FakeFeedback:
    """Minimal feedback-like object returned by MLflow scorers."""

    def __init__(self, value, rationale="some rationale"):
        self.value = value
        self.rationale = rationale


# ---------------------------------------------------------------------------
# BEFORE / VALIDATION
# ---------------------------------------------------------------------------


class TestBeforeValidation:
    def test_pass(self):
        scorer = _mock_scorer(_FakeFeedback(value=True))
        guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
        req = _make_request()
        result = guard.process_request(req)
        assert result is req
        scorer.assert_called_once_with(outputs="Hello, world!")

    def test_block(self):
        scorer = _mock_scorer(_FakeFeedback(value=False, rationale="toxic content"))
        guard = JudgeGuardrail(
            scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION, name="safety"
        )
        with pytest.raises(GuardrailViolation, match="safety.*toxic content"):
            guard.process_request(_make_request())
        scorer.assert_called_once()

    def test_skip_response(self):
        scorer = _mock_scorer(_FakeFeedback(value=False))
        guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
        resp = _make_response()
        result = guard.process_response(resp)
        assert result is resp
        scorer.assert_not_called()


# ---------------------------------------------------------------------------
# AFTER / VALIDATION
# ---------------------------------------------------------------------------


class TestAfterValidation:
    def test_pass(self):
        scorer = _mock_scorer(_FakeFeedback(value="yes"))
        guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
        resp = _make_response()
        result = guard.process_response(resp)
        assert result is resp
        scorer.assert_called_once_with(outputs="I'm a helpful assistant.")

    def test_block(self):
        scorer = _mock_scorer(_FakeFeedback(value="no", rationale="PII detected"))
        guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION, name="pii")
        with pytest.raises(GuardrailViolation, match="pii.*PII detected"):
            guard.process_response(_make_response())
        scorer.assert_called_once()

    def test_skip_request(self):
        scorer = _mock_scorer(_FakeFeedback(value=False))
        guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
        req = _make_request()
        result = guard.process_request(req)
        assert result is req
        scorer.assert_not_called()


# ---------------------------------------------------------------------------
# SANITIZATION (currently raises)
# ---------------------------------------------------------------------------


class TestSanitization:
    def test_before_sanitization_raises_on_fail(self):
        scorer = _mock_scorer(_FakeFeedback(value=False, rationale="needs cleaning"))
        guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION)
        with pytest.raises(GuardrailViolation, match="Sanitization not yet implemented"):
            guard.process_request(_make_request())

    def test_after_sanitization_raises_on_fail(self):
        scorer = _mock_scorer(_FakeFeedback(value=False, rationale="redact"))
        guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.SANITIZATION)
        with pytest.raises(GuardrailViolation, match="Sanitization not yet implemented"):
            guard.process_response(_make_response())

    def test_sanitization_passes_on_good_content(self):
        scorer = _mock_scorer(_FakeFeedback(value=True))
        guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.SANITIZATION)
        req = _make_request()
        assert guard.process_request(req) is req


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_pass"),
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("true", True),
        ("pass", True),
        ("no", False),
        ("false", False),
        ("fail", False),
        (1, True),
        (0, False),
    ],
)
def test_is_passing_various_values(value, expected_pass):
    scorer = _mock_scorer(_FakeFeedback(value=value))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    if expected_pass:
        result = guard.process_request(_make_request())
        assert result is not None
    else:
        with pytest.raises(GuardrailViolation, match="blocked"):
            guard.process_request(_make_request())


def test_empty_messages_request():
    scorer = _mock_scorer(_FakeFeedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    result = guard.process_request({"messages": []})
    assert result == {"messages": []}
    scorer.assert_called_once_with(outputs="")


def test_empty_choices_response():
    scorer = _mock_scorer(_FakeFeedback(value=True))
    guard = JudgeGuardrail(scorer, GuardrailStage.AFTER, GuardrailAction.VALIDATION)
    result = guard.process_response({"choices": []})
    assert result == {"choices": []}
    scorer.assert_called_once_with(outputs="")


def test_plain_value_result():
    scorer = _mock_scorer(True)
    guard = JudgeGuardrail(scorer, GuardrailStage.BEFORE, GuardrailAction.VALIDATION)
    result = guard.process_request(_make_request())
    assert result is not None
