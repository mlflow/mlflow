"""Tests for SIMBAAlignmentOptimizer."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

#ALKIS: remove this test. It tests private attributes.
def test_simba_optimizer_instantiation():
    """Test that SIMBAAlignmentOptimizer can be instantiated with default parameters."""
    optimizer = SIMBAAlignmentOptimizer()
    assert optimizer._bsize == SIMBAAlignmentOptimizer.DEFAULT_BSIZE
    assert optimizer._seed == SIMBAAlignmentOptimizer.DEFAULT_SEED
    assert hasattr(optimizer, "_logger")

#ALKIS: remove this test. It tests private attributes.
def test_simba_optimizer_class_constants():
    """Test that class constants are properly defined."""
    assert SIMBAAlignmentOptimizer.DEFAULT_BSIZE == 4
    assert SIMBAAlignmentOptimizer.DEFAULT_SEED == 42

    # Test that defaults are used
    optimizer = SIMBAAlignmentOptimizer()
    assert optimizer._bsize == SIMBAAlignmentOptimizer.DEFAULT_BSIZE
    assert optimizer._seed == SIMBAAlignmentOptimizer.DEFAULT_SEED

#ALKIS: remove this test. It tests private attributes.
def test_simba_optimizer_simplified_constructor():
    """Test SIMBAAlignmentOptimizer simplified constructor with defaults only."""
    optimizer = SIMBAAlignmentOptimizer()
    assert optimizer._bsize == 4
    assert optimizer._seed == 42

#ALKIS: remove this test. It tests private attributes.
def test_dspy_optimize_parameters():
    """Test that SIMBA optimizer has the correct parameters."""
    optimizer = SIMBAAlignmentOptimizer()

    # Test parameters are stored correctly
    assert optimizer._bsize == 4
    assert optimizer._seed == 42

    # Test _dspy_optimize method exists
    assert hasattr(optimizer, "_dspy_optimize")
    assert callable(optimizer._dspy_optimize)


def test_dspy_optimize_no_dspy():
    """Test optimization when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = SIMBAAlignmentOptimizer()

        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer._dspy_optimize(Mock(), [], Mock())


# ALKIS: refactor this test into a method that we can resuse to test other optimizers. 
def test_full_alignment_workflow(mock_judge, sample_traces_with_assessments):
    """Test complete alignment workflow with SIMBA."""
    mock_dspy = MagicMock()
    # Setup DSPy mocks
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    mock_signature = Mock()
    mock_dspy.make_signature.return_value = mock_signature

    mock_simba = Mock()
    mock_compiled_program = Mock()
    mock_compiled_program.signature.instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    mock_simba.compile.return_value = mock_compiled_program
    mock_dspy.SIMBA.return_value = mock_simba
    mock_dspy.LM.return_value = Mock()

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        # # Patch make_judge where it's imported in the dspy module
        # with patch("mlflow.genai.judges.optimizers.dspy.make_judge") as mock_make_judge:
        #     # Mock the optimized judge
        #     mock_optimized_judge = Mock()
        #     mock_optimized_judge.name = "mock_judge"
        #     mock_make_judge.return_value = mock_optimized_judge

        optimizer = SIMBAAlignmentOptimizer()
        result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result.model == mock_judge.model
    assert result.description == "Optimized instructions with {{inputs}} and {{outputs}}"

    # Verify DSPy components were called through the optimization process
    assert mock_dspy.SIMBA.called
    assert mock_dspy.make_signature.called
    

def test_alignment_with_default_parameters(mock_judge, sample_traces_with_assessments):
    """Test alignment with default SIMBA parameters."""
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
    mock_compiled_program.signature.instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    mock_simba.compile.return_value = mock_compiled_program
    mock_dspy.SIMBA.return_value = mock_simba
    mock_dspy.LM.return_value = Mock()

    # Test with default parameters
    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        # Patch make_judge where it's imported in the dspy module
        with patch("mlflow.genai.judges.optimizers.dspy.make_judge") as mock_make_judge:
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge"
            mock_make_judge.return_value = mock_optimized_judge

            optimizer = SIMBAAlignmentOptimizer()

            result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return a judge
    assert result is not None
    assert result == mock_optimized_judge

    # Verify DSPy components were called with alignment process
    assert mock_dspy.SIMBA.called
