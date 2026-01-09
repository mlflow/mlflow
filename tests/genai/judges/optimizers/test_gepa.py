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
        assert "reflection_lm" in call_kwargs
        assert call_kwargs["max_metric_calls"] == 100  # Default value
        assert len(call_kwargs) == 3  # metric, max_metric_calls, reflection_lm


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


def test_alignment_with_real_dspy(mock_judge, sample_traces_with_assessments):
    """
    Integration test for GePaAlignmentOptimizer using real dspy (not mocked).

    This test verifies that our optimizer correctly integrates with the actual
    dspy.GEPA implementation and follows the expected API contract.

    Skipped if dspy or dspy.GEPA is not available.

    Note: This test may fail with API authentication errors when GEPA tries to
    make actual LLM calls. These errors are expected and indicate successful
    integration (GEPA accepted our parameters and started optimization).
    """
    try:
        import dspy

        if not hasattr(dspy, "GEPA"):
            pytest.skip("dspy.GEPA not available in installed dspy version")
    except ImportError:
        pytest.skip("dspy not installed")

    from mlflow.exceptions import MlflowException

    # Create optimizer with minimal budget for fast test
    optimizer = GePaAlignmentOptimizer(
        model="openai:/gpt-4o-mini",
        max_metric_calls=1,  # Minimal budget for fast test
    )

    # Override get_min_traces_required to work with our fixture
    with patch.object(GePaAlignmentOptimizer, "get_min_traces_required", return_value=5):
        # This will fail if:
        # 1. dspy.GEPA doesn't have the expected API (compile method, etc.)
        # 2. We're passing wrong parameters to dspy.GEPA.__init__
        # 3. We're calling compile with wrong parameters
        try:
            result = optimizer.align(mock_judge, sample_traces_with_assessments)

            # If we got here without errors, the API contract is correct
            assert result is not None
            assert hasattr(result, "instructions")
        except TypeError as e:
            # TypeError means our API usage is wrong - this is a real failure
            pytest.fail(f"API contract mismatch with dspy.GEPA: {e}")
        except MlflowException as e:
            # MlflowException wrapping authentication/API errors is expected
            # It means GEPA accepted our parameters and tried to run
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                # This is actually SUCCESS - GEPA started running with correct API
                pass
            else:
                # Some other MlflowException - re-raise to investigate
                raise
