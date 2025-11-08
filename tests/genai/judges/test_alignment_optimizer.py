from unittest.mock import Mock, patch

import pytest

from mlflow.entities.trace import Trace
from mlflow.genai.judges import AlignmentOptimizer, Judge
from mlflow.genai.judges.base import JudgeField


class MockJudge(Judge):
    """Mock Judge implementation for testing."""

    def __init__(self, name: str = "mock_judge", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def instructions(self) -> str:
        return f"Mock judge implementation: {self.name}"

    def get_input_fields(self) -> list[JudgeField]:
        """Get input fields for mock judge."""
        return [
            JudgeField(name="input", description="Mock input field"),
            JudgeField(name="output", description="Mock output field"),
        ]

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


def create_mock_traces():
    """Create mock traces for testing."""
    # Create minimal mock traces - just enough to pass type checking
    mock_trace = Mock(spec=Trace)
    return [mock_trace]


def test_judge_align_method():
    """Test the Judge.align convenience method."""
    judge = MockJudge(name="test_judge")
    optimizer = MockOptimizerWithTracking()
    # Replace the align method with a Mock to use built-in mechanisms
    optimizer.align = Mock(return_value=MockJudge(name="test_judge_aligned"))
    traces = create_mock_traces()

    optimized = judge.align(traces, optimizer=optimizer)

    # Verify the result
    assert isinstance(optimized, Judge)
    assert optimized.name == "test_judge_aligned"

    # Assert that optimizer.align was called with correct parameters using Mock's mechanisms
    optimizer.align.assert_called_once_with(judge, traces)


def test_judge_align_method_delegation():
    """Test that Judge.align properly delegates to optimizer.align."""
    judge = MockJudge()

    # Create a spy optimizer that records calls
    optimizer = Mock(spec=AlignmentOptimizer)
    expected_result = MockJudge(name="expected")
    optimizer.align.return_value = expected_result

    traces = create_mock_traces()

    result = judge.align(traces, optimizer=optimizer)

    # Verify delegation
    optimizer.align.assert_called_once_with(judge, traces)
    assert result is expected_result


def test_judge_align_with_default_optimizer():
    """Test that Judge.align uses default SIMBA optimizer when optimizer=None."""
    judge = MockJudge()
    traces = create_mock_traces()

    # Mock the get_default_optimizer function to return our mock
    expected_result = MockJudge(name="aligned_with_default")
    mock_optimizer = Mock(spec=AlignmentOptimizer)
    mock_optimizer.align.return_value = expected_result

    with patch("mlflow.genai.judges.base.get_default_optimizer", return_value=mock_optimizer):
        result = judge.align(traces)

    # Verify delegation to default optimizer
    mock_optimizer.align.assert_called_once_with(judge, traces)
    assert result is expected_result
