from importlib import reload
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer


def test_dspy_optimize_no_dspy():
    # Since dspy import is now at module level, we need to test this differently
    # The error should be raised when importing the module, not when calling methods

    def _reload_module():
        import mlflow.genai.judges.optimizers.simba as simba_module

        reload(simba_module)

    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(MlflowException, match="DSPy library is required"):
            _reload_module()


def test_full_alignment_workflow(mock_judge, sample_traces_with_assessments):
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


def test_custom_simba_parameters(mock_judge, sample_traces_with_assessments):
    mock_simba = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_simba.compile.return_value = mock_compiled_program

    def custom_metric(example, pred, trace=None):
        return True

    custom_batch_size = 15
    with patch("dspy.SIMBA") as mock_simba_class, patch("dspy.LM", MagicMock()):
        mock_simba_class.return_value = mock_simba
        optimizer = SIMBAAlignmentOptimizer(
            batch_size=custom_batch_size,
            seed=123,
            simba_kwargs={
                "metric": custom_metric,
                "max_demos": 5,
                "num_threads": 2,
                "max_steps": 10,
            },
        )
        with patch.object(SIMBAAlignmentOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

        # Verify SIMBA was initialized with custom parameters
        mock_simba_class.assert_called_once()
        call_kwargs = mock_simba_class.call_args.kwargs
        assert call_kwargs["bsize"] == custom_batch_size
        assert call_kwargs["metric"] == custom_metric
        assert call_kwargs["max_demos"] == 5
        assert call_kwargs["num_threads"] == 2
        assert call_kwargs["max_steps"] == 10

        # Verify seed is passed to compile
        mock_simba.compile.assert_called_once()
        compile_kwargs = mock_simba.compile.call_args.kwargs
        assert compile_kwargs["seed"] == 123


def test_default_parameters_not_passed(mock_judge, sample_traces_with_assessments):
    mock_simba = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_simba.compile.return_value = mock_compiled_program

    with patch("dspy.SIMBA") as mock_simba_class, patch("dspy.LM", MagicMock()):
        mock_simba_class.return_value = mock_simba
        optimizer = SIMBAAlignmentOptimizer()
        with patch.object(SIMBAAlignmentOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

        # Verify only required parameters are passed
        mock_simba_class.assert_called_once()
        call_kwargs = mock_simba_class.call_args.kwargs
        assert "metric" in call_kwargs
        assert "bsize" in call_kwargs
        assert len(call_kwargs) == 2
