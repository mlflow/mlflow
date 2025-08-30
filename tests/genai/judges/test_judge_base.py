from typing import Any

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.judges import Judge
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.scorers.base import Scorer


class MockJudgeImplementation(Judge):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def model(self) -> str:
        return "mock://test-model"

    @property
    def instructions(self) -> str:
        return "Mock judge instructions for testing"

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(name="inputs", description="Test inputs"),
            JudgeField(name="outputs", description="Test outputs"),
        ]

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
    with pytest.raises(TypeError, match="Can't instantiate abstract class Judge"):
        Judge(name="test")


def test_judge_implementation():
    judge = MockJudgeImplementation(name="test_judge")

    assert isinstance(judge, Scorer)
    assert isinstance(judge, Judge)

    result = judge(
        inputs={"question": "What is 2+2?"},
        outputs="4",
    )
    assert isinstance(result, Feedback)
    assert result.name == "test_judge"
    assert result.value is True
    assert "Test evaluation by test_judge" in result.rationale


def test_judge_factory_pattern():
    def make_simple_judge(name: str) -> Judge:
        class DynamicJudge(Judge):
            @property
            def model(self) -> str:
                return "mock://dynamic-model"

            @property
            def instructions(self) -> str:
                return "Dynamic judge instructions"

            def get_input_fields(self) -> list[JudgeField]:
                return [
                    JudgeField(name="inputs", description="Dynamic test inputs"),
                    JudgeField(name="outputs", description="Dynamic test outputs"),
                ]

            def __call__(self, **kwargs):
                return Feedback(name=self.name, value="pass", rationale=f"Evaluated by {self.name}")

        return DynamicJudge(name=name)

    judge = make_simple_judge(name="factory_judge")

    assert isinstance(judge, Judge)
    assert isinstance(judge, Scorer)
    assert judge.name == "factory_judge"

    result = judge(outputs="test output")
    assert isinstance(result, Feedback)
    assert result.value == "pass"
    assert "Evaluated by factory_judge" in result.rationale
