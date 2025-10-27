from typing import Any

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.judges import Judge
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.scorers.base import Scorer


class MockJudgeImplementation(Judge):
    def __init__(self, name: str, custom_instructions: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._custom_instructions = custom_instructions

    @property
    def instructions(self) -> str:
        if self._custom_instructions:
            return self._custom_instructions
        return f"Mock judge implementation: {self.name}"

    def get_input_fields(self) -> list[JudgeField]:
        """Get input fields for mock judge."""
        return [
            JudgeField(name="inputs", description="Input data for evaluation"),
            JudgeField(name="outputs", description="Output data for evaluation"),
            JudgeField(name="expectations", description="Expected outcomes"),
            JudgeField(name="trace", description="Trace for evaluation"),
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
    assert judge.instructions == "Mock judge implementation: test_judge"

    result = judge(
        inputs={"question": "What is 2+2?"},
        outputs="4",
    )
    assert isinstance(result, Feedback)
    assert result.name == "test_judge"
    assert result.value is True
    assert "Test evaluation by test_judge" in result.rationale

    judge_custom = MockJudgeImplementation(
        name="custom_judge", custom_instructions="Custom instructions for testing"
    )
    assert judge_custom.instructions == "Custom instructions for testing"


def test_judge_factory_pattern():
    def make_simple_judge(name: str, instructions: str) -> Judge:
        class DynamicJudge(Judge):
            @property
            def instructions(self) -> str:
                return instructions

            def get_input_fields(self) -> list[JudgeField]:
                """Get input fields for dynamic judge."""
                return [
                    JudgeField(name="outputs", description="Output to evaluate"),
                ]

            def __call__(self, **kwargs):
                return Feedback(name=self.name, value="pass", rationale=f"Evaluated by {self.name}")

        return DynamicJudge(name=name)

    judge = make_simple_judge(
        name="factory_judge", instructions="A judge created by factory function"
    )

    assert isinstance(judge, Judge)
    assert isinstance(judge, Scorer)
    assert judge.name == "factory_judge"
    assert judge.instructions == "A judge created by factory function"

    result = judge(outputs="test output")
    assert isinstance(result, Feedback)
    assert result.value == "pass"
    assert "Evaluated by factory_judge" in result.rationale
