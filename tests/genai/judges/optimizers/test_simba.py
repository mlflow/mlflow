"""Tests for SIMBAAlignmentOptimizer."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

def test_dspy_optimize_no_dspy():
    """Test optimization when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = SIMBAAlignmentOptimizer()

        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer._dspy_optimize(Mock(), [], Mock())


def test_full_alignment_workflow(mock_judge, sample_traces_with_assessments):
    """Test complete alignment workflow with SIMBA."""
    mock_simba = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    mock_simba.compile.return_value = mock_compiled_program

    with patch("dspy.SIMBA", MagicMock()) as mock_simba_class, patch("dspy.LM", MagicMock()):
        mock_simba_class.return_value = mock_simba
        optimizer = SIMBAAlignmentOptimizer()
        result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result.model == mock_judge.model
    assert result.instructions == "Optimized instructions with {{inputs}} and {{outputs}}"    