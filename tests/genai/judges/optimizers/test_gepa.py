from importlib import reload
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers import GePaAlignmentOptimizer


def test_dspy_optimize_no_dspy():
    # Since dspy import is now at module level, we need to test this differently
    # The error should be raised when importing the module, not when calling methods

    def _reload_module():
        import mlflow.genai.judges.optimizers.gepa as gepa_module

        reload(gepa_module)

    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(MlflowException, match="DSPy library is required"):
            _reload_module()


def test_full_alignment_workflow(mock_judge, sample_traces_with_assessments):
    mock_gepa = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_gepa.compile.return_value = mock_compiled_program

    with (
        patch("dspy.GEPA", MagicMock(), create=True) as mock_gepa_class,
        patch("dspy.LM", MagicMock()),
    ):
        mock_gepa_class.return_value = mock_gepa
        optimizer = GePaAlignmentOptimizer()
        # Mock get_min_traces_required to work with 5 traces from fixture
        with patch.object(GePaAlignmentOptimizer, "get_min_traces_required", return_value=5):
            result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result.model == mock_judge.model
    # The judge instructions should be the raw optimized instructions
    expected_instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    assert result.instructions == expected_instructions


def test_custom_gepa_parameters(mock_judge, sample_traces_with_assessments):
    mock_gepa = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_gepa.compile.return_value = mock_compiled_program

    def custom_metric(example, pred, trace=None):
        return True

    with patch("dspy.GEPA", create=True) as mock_gepa_class, patch("dspy.LM", MagicMock()):
        mock_gepa_class.return_value = mock_gepa
        optimizer = GePaAlignmentOptimizer(
            max_metric_calls=50,
            gepa_kwargs={
                "metric": custom_metric,
                "candidate_pool_size": 10,
                "num_threads": 4,
            },
        )
        with patch.object(GePaAlignmentOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

        # Verify GEPA was initialized with custom parameters
        mock_gepa_class.assert_called_once()
        call_kwargs = mock_gepa_class.call_args.kwargs
        assert call_kwargs["max_metric_calls"] == 50
        assert call_kwargs["metric"] == custom_metric
        assert call_kwargs["candidate_pool_size"] == 10
        assert call_kwargs["num_threads"] == 4


def test_default_parameters(mock_judge, sample_traces_with_assessments):
    mock_gepa = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_gepa.compile.return_value = mock_compiled_program

    with patch("dspy.GEPA", create=True) as mock_gepa_class, patch("dspy.LM", MagicMock()):
        mock_gepa_class.return_value = mock_gepa
        optimizer = GePaAlignmentOptimizer()
        with patch.object(GePaAlignmentOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

        # Verify only required parameters are passed with defaults
        mock_gepa_class.assert_called_once()
        call_kwargs = mock_gepa_class.call_args.kwargs
        assert "metric" in call_kwargs
        assert "max_metric_calls" in call_kwargs
        assert call_kwargs["max_metric_calls"] == 100  # Default value
        assert len(call_kwargs) == 2


def test_gepa_kwargs_override_defaults(mock_judge, sample_traces_with_assessments):
    mock_gepa = MagicMock()
    mock_compiled_program = MagicMock()
    mock_compiled_program.signature = MagicMock()
    mock_compiled_program.signature.instructions = (
        "Optimized instructions with {{inputs}} and {{outputs}}"
    )
    mock_gepa.compile.return_value = mock_compiled_program

    def custom_metric(example, pred, trace=None):
        return True

    with patch("dspy.GEPA", create=True) as mock_gepa_class, patch("dspy.LM", MagicMock()):
        mock_gepa_class.return_value = mock_gepa
        optimizer = GePaAlignmentOptimizer(
            max_metric_calls=30,
            gepa_kwargs={
                "metric": custom_metric,  # This should override the default metric
            },
        )
        with patch.object(GePaAlignmentOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

        # Verify the custom metric from gepa_kwargs overrides the default
        mock_gepa_class.assert_called_once()
        call_kwargs = mock_gepa_class.call_args.kwargs
        assert call_kwargs["metric"] == custom_metric
        assert call_kwargs["max_metric_calls"] == 30
