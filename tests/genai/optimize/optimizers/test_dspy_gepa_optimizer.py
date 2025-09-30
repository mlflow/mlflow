import sys
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("dspy", minversion="2.6.0")

from mlflow import register_prompt
from mlflow.genai.optimize.optimizers import _DSPyGEPAOptimizer
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig, OptimizerOutput
from mlflow.genai.scorers import scorer


@pytest.fixture
def mock_gepa():
    with patch("dspy.GEPA") as mock:
        yield mock


@pytest.fixture
def mock_extractor():
    with patch(
        "mlflow.genai.optimize.optimizers._DSPyGEPAOptimizer._extract_instructions",
        new=lambda self, template, lm: template,
    ) as mock:
        yield mock


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "inputs": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
            ],
            "expectations": [{"translation": "Hola"}, {"translation": "Monde"}],
        }
    )


@pytest.fixture
def sample_prompt():
    return register_prompt(
        name="test_gepa_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
    )


@scorer
def sample_scorer(inputs, outputs, expectations):
    return 1.0


@pytest.mark.parametrize(
    "use_eval_data",
    [
        False,
        True,
    ],
)
def test_optimize_scenarios(
    mock_gepa,
    sample_data,
    sample_prompt,
    mock_extractor,
    capsys,
    use_eval_data,
):
    import dspy

    optimizer = _DSPyGEPAOptimizer(OptimizerConfig())

    optimized_program = dspy.Predict("input_text, language -> translation")
    optimized_program.score = 1.0

    # Mock detailed_results with val_aggregate_scores
    class MockDetailedResults:
        val_aggregate_scores = [0.5, 0.7, 0.9, 1.0]

    optimized_program.detailed_results = MockDetailedResults()
    mock_gepa.return_value.compile.return_value = optimized_program

    # Prepare eval_data if needed
    eval_data = sample_data.copy() if use_eval_data else None

    result = optimizer.optimize(
        prompt=sample_prompt,
        target_llm_params=LLMParams(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
        eval_data=eval_data,
    )

    # Verify reflection LM settings
    mock_gepa.assert_called_once()
    kwargs = mock_gepa.call_args[1]
    assert "reflection_lm" in kwargs

    # Verify GEPA configuration
    assert "auto" in kwargs
    assert "track_stats" in kwargs
    assert kwargs["track_stats"] is True

    # Verify optimization result
    assert isinstance(result, OptimizerOutput)
    assert result.final_eval_score == optimized_program.score
    assert result.initial_eval_score == 0.5
    assert result.optimizer_name == "DSPy/GEPA"

    # Verify eval data handling
    compile_args = mock_gepa.return_value.compile.call_args[1]
    assert "trainset" in compile_args
    assert "valset" in compile_args
    if eval_data is not None:
        assert compile_args["valset"] is not None
    else:
        assert compile_args["valset"] is None

    # Verify logging
    captured = capsys.readouterr()
    assert "Starting prompt optimization" in captured.err
    assert "Optimization complete! Initial score: 0.5. Final score: 1.0" in captured.err


@pytest.mark.parametrize(
    "verbose",
    [
        False,
        True,
    ],
)
def test_optimize_with_verbose(
    mock_gepa, sample_data, sample_prompt, mock_extractor, verbose, capsys
):
    import dspy

    def compile_side_effect(*args, **kwargs):
        print("GEPA optimization progress")  # noqa: T201
        print("GEPA debug info", file=sys.stderr)  # noqa: T201
        return dspy.Predict("input_text, language -> translation")

    mock_gepa.return_value.compile.side_effect = compile_side_effect
    optimizer = _DSPyGEPAOptimizer(OptimizerConfig(verbose=verbose))

    optimizer.optimize(
        prompt=sample_prompt,
        target_llm_params=LLMParams(model_name="agent/model"),
        train_data=sample_data,
        scorers=[sample_scorer],
    )

    captured = capsys.readouterr()
    if verbose:
        assert "GEPA optimization progress" in captured.out
        assert "GEPA debug info" in captured.err
    else:
        assert "GEPA optimization progress" not in captured.out
        assert "GEPA debug info" not in captured.err

    mock_gepa.assert_called_once()


def test_extract_eval_scores_with_detailed_results():
    import dspy

    optimizer = _DSPyGEPAOptimizer(OptimizerConfig())

    program = dspy.Predict("input -> output")

    # Mock detailed_results with val_aggregate_scores
    class MockDetailedResults:
        val_aggregate_scores = [0.5, 0.7, 0.9, 0.85]

    program.detailed_results = MockDetailedResults()

    initial_score, final_score = optimizer._extract_eval_scores(program)

    assert final_score == 0.9  # max of val_aggregate_scores
    assert initial_score == 0.5  # first score in val_aggregate_scores


@pytest.mark.parametrize(
    ("initial_score", "final_score", "expected_messages"),
    [
        # Test stable score
        (0.8, 0.8, ["Score remained stable at: 0.8"]),
        # Test improved score
        (0.5, 0.9, ["Initial score: 0.5", "Final score: 0.9", "+0.4000"]),
        # Test no initial score
        (None, 0.9, ["Final score: 0.9"]),
    ],
)
def test_display_optimization_result(capsys, initial_score, final_score, expected_messages):
    optimizer = _DSPyGEPAOptimizer(OptimizerConfig())

    optimizer._display_optimization_result(initial_score, final_score)

    captured = capsys.readouterr()
    for message in expected_messages:
        assert message in captured.err

    # For no initial score case, verify "Initial score" is not in output
    if initial_score is None:
        assert "Initial score" not in captured.err
