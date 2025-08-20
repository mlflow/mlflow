from unittest.mock import patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges import Judge
from mlflow.genai.scorers.base import ScorerKind


def test_judge_initialization():
    judge = Judge(
        name="test_judge",
        instructions="Check if output is valid",
        model="openai/gpt-4o-mini",
    )

    assert judge.name == "test_judge"
    assert judge.instructions == "Check if output is valid"
    assert judge.model == "openai/gpt-4o-mini"
    assert judge._examples == []
    assert judge.description == "Check if output is valid"


def test_judge_with_examples():
    examples = [
        {"inputs": {"q": "test"}, "outputs": "answer", "assessment": True},
        {"inputs": {"q": "test2"}, "outputs": "answer2", "assessment": False},
    ]

    judge = Judge(
        name="test_judge",
        instructions="Check validity",
        model="openai/gpt-4",
        examples=examples,
    )

    assert judge._examples == examples


def test_judge_kind_is_builtin():
    judge = Judge(
        name="test_judge",
        instructions="Test instructions",
        model="openai/gpt-4",
    )

    assert judge.kind == ScorerKind.BUILTIN


def test_judge_align_not_implemented():
    judge = Judge(
        name="test_judge",
        instructions="Test instructions",
        model="openai/gpt-4",
    )

    with pytest.raises(NotImplementedError, match="alignment is not yet implemented"):
        judge.align([])


def test_judge_call_invokes_model():
    with patch("mlflow.genai.judges.utils.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = Feedback(name="test_judge", value=True, rationale="Test passed")

        judge = Judge(
            name="test_judge",
            instructions="Check if output is valid",
            model="openai/gpt-4",
        )

        result = judge(
            inputs={"question": "What is 2+2?"},
            outputs="4",
        )

        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        assert call_args[1]["model"] == "openai/gpt-4"
        assert call_args[1]["name"] == "test_judge"
        assert call_args[1]["prompt"] == "Check if output is valid"

        assert isinstance(result, Feedback)
        assert result.name == "test_judge"
        assert result.value is True
        assert result.rationale == "Test passed"


def test_judge_call_converts_primitive_to_feedback():
    with patch("mlflow.genai.judges.utils.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = "pass"

        judge = Judge(
            name="test_judge",
            instructions="Check validity",
            model="openai/gpt-4",
        )

        result = judge(outputs="test output")

        assert isinstance(result, Feedback)
        assert result.name == "test_judge"
        assert result.value == "pass"
        assert "Evaluated by test_judge judge using openai/gpt-4" in result.rationale
