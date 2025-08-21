from typing import Any

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.judges import Judge
from mlflow.genai.scorers.base import Scorer


class MockJudgeImplementation(Judge):
    def __init__(self, name: str, custom_description: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._custom_description = custom_description

    @property
    def description(self) -> str:
        if self._custom_description:
            return self._custom_description
        return f"Mock judge implementation: {self.name}"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        return Feedback(
            name=self.name,
            value=True,
            rationale=f"Test evaluation by {self.name}",
        )


def test_judge_base_class_abstract_behavior():
    judge = Judge(name="test")

    with pytest.raises(NotImplementedError, match="Implementation of __call__ is required"):
        judge(outputs="test")

    with pytest.raises(NotImplementedError, match="Judge.description must be implemented"):
        _ = judge.description

    assert judge.name == "test"


def test_judge_implementation():
    judge = MockJudgeImplementation(name="test_judge")

    assert isinstance(judge, Scorer)
    assert isinstance(judge, Judge)
    assert judge.description == "Mock judge implementation: test_judge"

    result = judge(
        inputs={"question": "What is 2+2?"},
        outputs="4",
    )
    assert isinstance(result, Feedback)
    assert result.name == "test_judge"
    assert result.value is True
    assert "Test evaluation by test_judge" in result.rationale

    judge_custom = MockJudgeImplementation(
        name="custom_judge", custom_description="Custom description for testing"
    )
    assert judge_custom.description == "Custom description for testing"


def test_judge_factory_pattern():
    def make_simple_judge(name: str, description: str) -> Judge:
        class DynamicJudge(Judge):
            @property
            def description(self) -> str:
                return description

            def __call__(self, **kwargs):
                return Feedback(name=self.name, value="pass", rationale=f"Evaluated by {self.name}")

        return DynamicJudge(name=name)

    judge = make_simple_judge(
        name="factory_judge", description="A judge created by factory function"
    )

    assert isinstance(judge, Judge)
    assert isinstance(judge, Scorer)
    assert judge.name == "factory_judge"
    assert judge.description == "A judge created by factory function"

    result = judge(outputs="test output")
    assert isinstance(result, Feedback)
    assert result.value == "pass"
    assert "Evaluated by factory_judge" in result.rationale
