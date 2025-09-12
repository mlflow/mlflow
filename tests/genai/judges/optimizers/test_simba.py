"""Tests for SIMBAAlignmentOptimizer."""

from importlib import reload
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer


def test_dspy_optimize_no_dspy():
    """Test that SIMBAAlignmentOptimizer raises error when DSPy is not available."""
    # Since dspy import is now at module level, we need to test this differently
    # The error should be raised when importing the module, not when calling methods

    def _reload_module():
        import mlflow.genai.judges.optimizers.simba as simba_module

        reload(simba_module)

    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(MlflowException, match="DSPy library is required"):
            _reload_module()


def test_full_alignment_workflow(mock_judge, sample_traces_with_assessments):
    """Test complete alignment workflow with SIMBA."""
    mock_simba = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_simba.compile.return_value = mock_compiled_program

    with patch("dspy.SIMBA", MagicMock()) as mock_simba_class, patch("dspy.LM", MagicMock()):
        mock_simba_class.return_value = mock_simba
        optimizer = SIMBAAlignmentOptimizer()
        # Mock get_min_traces_required to work with 5 traces from fixture
        with patch.object(SIMBAAlignmentOptimizer, "get_min_traces_required", return_value=5):
            result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result.model == mock_judge.model
    # The judge instructions should be the raw optimized instructions
    expected_instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    assert result.instructions == expected_instructions
