from unittest.mock import Mock

import pytest

from mlflow.entities.trace import Trace
from mlflow.genai.judges import AlignmentOptimizer, Judge


class MockJudge(Judge):
    """Mock Judge implementation for testing."""

    def __init__(self, name: str = "mock_judge", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def instructions(self) -> str:
        return "Mock judge instructions"

    @property
    def model(self) -> str:
        return "mock-model"

    def get_input_fields(self):
        from mlflow.genai.judges.base import JudgeField

        return [JudgeField(name="inputs", description="Mock inputs")]

    def __call__(self, **kwargs):
        from mlflow.entities.assessment import Feedback

        return Feedback(name=self.name, value=True, rationale="Mock evaluation")


class MockOptimizer(AlignmentOptimizer):
    """Mock AlignmentOptimizer implementation for testing."""

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        # Return a new judge with modified name to show it was processed
        return MockJudge(name=f"{judge.name}_optimized")


def test_alignment_optimizer_abstract():
    """Test that AlignmentOptimizer cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class AlignmentOptimizer"):
        AlignmentOptimizer()


def test_alignment_optimizer_align_method_required():
    """Test that concrete classes must implement align method."""

    class IncompleteOptimizer(AlignmentOptimizer):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteOptimizer"):
        IncompleteOptimizer()


def test_concrete_optimizer_implementation():
    """Test that concrete optimizer can be instantiated and used."""
    optimizer = MockOptimizer()
    judge = MockJudge(name="test_judge")
    traces = []  # Empty traces for testing

    # Should not raise any errors
    optimizer.align(judge, traces)


class MockOptimizerWithTracking(AlignmentOptimizer):
    """Mock AlignmentOptimizer implementation with call tracking for integration tests."""

    def __init__(self):
        self.align_called = False
        self.align_args = None

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        self.align_called = True
        self.align_args = (judge, traces)

        # Return a new judge with modified name to show it was processed
        return MockJudge(name=f"{judge.name}_aligned")


def test_judge_align_method_delegation():
    """Test that Judge.align properly delegates to optimizer.align."""
    judge = MockJudge()

    # Create a spy optimizer that records calls
    optimizer = Mock(spec=AlignmentOptimizer)
    expected_result = MockJudge(name="expected")
    optimizer.align.return_value = expected_result

    traces = [Mock(spec=Trace)]

    result = judge.align(optimizer, traces)

    # Verify delegation
    optimizer.align.assert_called_once_with(judge, traces)
    assert result is expected_result
