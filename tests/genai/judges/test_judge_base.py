from typing import Any

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.judges import Judge
from mlflow.genai.scorers.base import Scorer


class MockJudgeImplementation(Judge):
    def __init__(self, name: str, model: str, examples: list[dict[str, Any]] | None = None):
        super().__init__(name=name, model=model)
        if examples:
            self._examples = examples

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """Simple implementation for testing."""
        return Feedback(
            name=self.name,
            value=True,
            rationale=f"Test evaluation by {self.name}",
        )


def test_judge_call_is_abstract():
    judge = Judge(name="test", model="test-model")

    with pytest.raises(NotImplementedError, match="Implementation of __call__ is required"):
        judge(outputs="test")


def test_judge_implementation_inherits_from_scorer():
    judge = MockJudgeImplementation(name="test_judge", model="test-model")
    assert isinstance(judge, Scorer)
    assert isinstance(judge, Judge)


def test_judge_implementation_has_model_field():
    judge = MockJudgeImplementation(name="test_judge", model="openai/gpt-4")
    assert judge.model == "openai/gpt-4"
    assert judge.name == "test_judge"


def test_judge_implementation_with_examples():
    examples = [
        {"inputs": {"q": "test"}, "outputs": "answer", "assessment": True},
        {"inputs": {"q": "test2"}, "outputs": "answer2", "assessment": False},
    ]

    judge = MockJudgeImplementation(
        name="test_judge",
        model="openai/gpt-4",
        examples=examples,
    )

    assert judge._examples == examples


def test_judge_implementation_call_method():
    judge = MockJudgeImplementation(name="test_judge", model="test-model")

    result = judge(
        inputs={"question": "What is 2+2?"},
        outputs="4",
    )

    assert isinstance(result, Feedback)
    assert result.name == "test_judge"
    assert result.value is True
    assert "Test evaluation by test_judge" in result.rationale


def test_judge_base_class_is_minimal():
    assert "model" in Judge.model_fields

    judge = Judge(name="test", model="test-model")
    assert judge.model == "test-model"
