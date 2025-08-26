import pytest

from mlflow.entities.trace import Trace
from mlflow.genai.judges import AlignmentOptimizer, Judge


class MockJudge(Judge):
    """Mock Judge implementation for testing."""

    def __init__(self, name: str = "mock_judge", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def description(self) -> str:
        return f"Mock judge implementation: {self.name}"

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
    result = optimizer.align(judge, traces)

    assert isinstance(result, Judge)
    assert result.name == "test_judge_optimized"
