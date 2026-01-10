from importlib import reload
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers import GePaAlignmentOptimizer

from tests.genai.judges.optimizers.conftest import create_mock_judge_evaluator


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


def test_gepa_runs_without_authentication_errors(mock_judge, sample_traces_with_assessments):
    """
    End-to-end test verifying GEPA optimization runs without authentication errors.

    This test exercises the complete GEPA optimization loop using mocked components:
    - GEPA proposes new instructions via reflection_lm (DummyLM)
    - Metric evaluates proposals against human assessments
    - GEPA iterates through optimization attempts
    - Returns a valid judge without requiring API authentication

    Note: GEPA may not modify instructions in this test because the mocking
    prevents it from collecting valid predictions for reflection. The primary
    goal is to verify the integration works without authentication errors.

    Skipped if dspy or dspy.GEPA is not available.
    """
    try:
        import dspy
        from dspy.utils.dummies import DummyLM

        if not hasattr(dspy, "GEPA"):
            pytest.skip("dspy.GEPA not available in installed dspy version")
    except ImportError:
        pytest.skip("dspy not installed")

    from mlflow.genai.judges.base import Judge

    # Configure DummyLM with deterministic instruction proposals
    # GEPA will request new instructions during reflection phase
    dummy_lm = DummyLM(
        [
            "Carefully evaluate whether the {{outputs}} effectively addresses {{inputs}}",
            "Assess if the {{outputs}} properly responds to the {{inputs}} query",
            "Determine whether {{outputs}} satisfactorily answers {{inputs}}",
            "Judge if {{outputs}} adequately resolves {{inputs}}",
            "Evaluate the quality of {{outputs}} in addressing {{inputs}}",
        ]
    )

    # Create optimizer with minimal budget for fast test
    # Note: Using max_metric_calls=10 to give GEPA enough budget to actually
    # run optimization iterations and modify instructions
    optimizer = GePaAlignmentOptimizer(
        model="openai:/gpt-4o-mini",
        max_metric_calls=10,
    )

    # Use shared mock judge evaluator from conftest
    mock_invoke_judge_model = create_mock_judge_evaluator()

    # Run optimization with DummyLM context and mocked judge invocations
    with (
        dspy.context(lm=dummy_lm),
        patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            side_effect=mock_invoke_judge_model,
        ) as mock_invoke,
        patch.object(
            GePaAlignmentOptimizer, "get_min_traces_required", return_value=5
        ) as mock_min_traces,
    ):
        result = optimizer.align(mock_judge, sample_traces_with_assessments)

    mock_invoke.assert_called()
    mock_min_traces.assert_called_once_with()

    # Verify optimization completed without errors
    assert result is not None
    assert isinstance(result, Judge)
    assert result.name == mock_judge.name
    assert result.model == mock_judge.model

    # Verify instructions are valid
    assert result.instructions is not None
    assert len(result.instructions) > 0

    # Verify template variables are preserved in the result
    assert "{{inputs}}" in result.instructions
    assert "{{outputs}}" in result.instructions
