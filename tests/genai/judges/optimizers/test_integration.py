"""Integration tests for optimizer implementations."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers import DSPyAlignmentOptimizer, SIMBAAlignmentOptimizer


def test_optimizer_imports():
    """Test that optimizers can be imported correctly."""
    # Test direct imports
    # Test module-level imports
    from mlflow.genai.judges.optimizers import DSPyAlignmentOptimizer, SIMBAAlignmentOptimizer
    from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer as DirectDSPy
    from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer as DirectSIMBA

    # Verify they're the same classes
    assert DirectDSPy == DSPyAlignmentOptimizer
    assert DirectSIMBA == SIMBAAlignmentOptimizer


def test_optimizer_hierarchy():
    """Test optimizer class hierarchy."""
    optimizer = SIMBAAlignmentOptimizer()

    assert isinstance(optimizer, DSPyAlignmentOptimizer)
    assert hasattr(optimizer, "align")
    assert hasattr(optimizer, "_dspy_optimize")


def test_simba_optimizer_workflow(mock_judge, sample_traces_with_assessments):
    """Test that SIMBA optimizer completes the full workflow."""
    mock_dspy = MagicMock()
    # Setup DSPy mocks
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    mock_signature = Mock()
    mock_dspy.make_signature.return_value = mock_signature

    mock_program = Mock()
    mock_dspy.ChainOfThought.return_value = mock_program

    mock_simba = Mock()
    mock_compiled_program = Mock()
    mock_compiled_program.signature.instructions = "Optimized instructions"
    mock_simba.compile.return_value = mock_compiled_program
    mock_dspy.SIMBA.return_value = mock_simba
    mock_dspy.LM.return_value = Mock()

    # Test SIMBA optimizer
    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge_optimized"
            mock_make_judge.return_value = mock_optimized_judge
            
            simba_optimizer = SIMBAAlignmentOptimizer()
            result = simba_optimizer.align(mock_judge, sample_traces_with_assessments)

    # Verify successful completion
    assert result is not None
    assert mock_dspy.SIMBA.called


def test_multiple_optimizers_independence():
    """Test that multiple optimizer instances are independent."""
    optimizer1 = SIMBAAlignmentOptimizer()
    optimizer2 = SIMBAAlignmentOptimizer()

    # Both should have same default parameters but be different instances
    assert optimizer1._bsize == optimizer2._bsize
    assert optimizer1._seed == optimizer2._seed
    assert optimizer1 is not optimizer2


def test_judge_integration(mock_judge, sample_traces_with_assessments):
    """Test integration with Judge.align method."""
    mock_dspy = MagicMock()
    # Setup DSPy mocks
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example
    mock_dspy.make_signature.return_value = Mock()
    mock_dspy.ChainOfThought.return_value = Mock()

    mock_simba = Mock()
    mock_compiled_program = Mock()
    mock_compiled_program.signature.instructions = "Optimized instructions"
    mock_simba.compile.return_value = mock_compiled_program
    mock_dspy.SIMBA.return_value = mock_simba
    mock_dspy.LM.return_value = Mock()

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge_optimized"
            mock_make_judge.return_value = mock_optimized_judge
            
            optimizer = SIMBAAlignmentOptimizer()

            # Test using the judge's align method
            result = mock_judge.align(optimizer, sample_traces_with_assessments)

    # Should return a judge instance
    assert result is not None


def test_error_propagation(mock_judge):
    """Test that errors are properly propagated through the system."""
    optimizer = SIMBAAlignmentOptimizer()

    # Test with empty traces
    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(mock_judge, [])


def test_trace_processing_consistency(mock_judge, sample_traces_with_assessments):
    """Test that trace processing is consistent across optimizer types."""
    mock_dspy = MagicMock()
    # Setup DSPy mocks
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example
    mock_dspy.make_signature.return_value = Mock()
    mock_dspy.ChainOfThought.return_value = Mock()

    mock_simba = Mock()
    mock_compiled_program = Mock()
    mock_compiled_program.signature.instructions = "Optimized instructions"
    mock_simba.compile.return_value = mock_compiled_program
    mock_dspy.SIMBA.return_value = mock_simba
    mock_dspy.LM.return_value = Mock()

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge_optimized"
            mock_make_judge.return_value = mock_optimized_judge
            
            optimizer = SIMBAAlignmentOptimizer()

            # Process traces - should not raise exceptions
            result = optimizer.align(mock_judge, sample_traces_with_assessments)
            assert result is not None

    # Verify DSPy.Example was called for each valid trace
    assert mock_dspy.Example.call_count >= 1