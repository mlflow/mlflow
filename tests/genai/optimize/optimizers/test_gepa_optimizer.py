import sys
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from mlflow.genai import register_prompt
from mlflow.genai.optimize.optimizers import _GEPAOptimizer
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig, OptimizerOutput
from mlflow.genai.scorers import scorer


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


class MockGEPAAdapter:
    pass


@scorer
def sample_scorer(inputs, outputs, expectations):
    return 1.0


def test_gepa_optimizer_initialization():
    optimizer = _GEPAOptimizer(OptimizerConfig())
    assert optimizer is not None
    assert optimizer.optimizer_config is not None


def test_gepa_optimizer_requires_gepa_package(sample_data):
    optimizer = _GEPAOptimizer(OptimizerConfig())

    with patch.dict(sys.modules, {"gepa": None}):
        with pytest.raises(ImportError, match="GEPA is not installed"):
            optimizer.optimize(
                prompt=Mock(template="test"),
                target_llm_params=LLMParams(model_name="openai/gpt-4"),
                train_data=sample_data,
                scorers=[sample_scorer],
            )


def test_gepa_optimizer_basic_flow(sample_data, sample_prompt):
    # Mock GEPA module
    mock_gepa = MagicMock()
    mock_result = Mock()
    mock_result.best_candidate = {"prompt": "Optimized prompt template"}
    mock_result.val_aggregate_scores = [0.5, 0.8, 0.95]
    mock_gepa.optimize.return_value = mock_result

    mock_gepa.GEPAAdapter = MockGEPAAdapter

    optimizer = _GEPAOptimizer(OptimizerConfig())

    with patch.dict(sys.modules, {"gepa": mock_gepa}):
        result = optimizer.optimize(
            prompt=sample_prompt,
            target_llm_params=LLMParams(model_name="openai/gpt-4o-mini"),
            train_data=sample_data,
            scorers=[sample_scorer],
        )

    # Verify result
    assert isinstance(result, OptimizerOutput)
    assert result.optimized_prompt == "Optimized prompt template"
    assert result.final_eval_score == 0.95
    assert result.initial_eval_score == 0.5
    assert result.optimizer_name == "GEPA"

    # Verify GEPA was called
    mock_gepa.optimize.assert_called_once()
    call_kwargs = mock_gepa.optimize.call_args.kwargs
    assert call_kwargs["seed_candidate"] == {"prompt": sample_prompt.template}
    assert "adapter" in call_kwargs


def test_gepa_optimizer_with_eval_data(sample_data, sample_prompt):
    mock_gepa = MagicMock()
    mock_result = Mock()
    mock_result.best_candidate = {"prompt": "Optimized prompt"}
    mock_result.val_aggregate_scores = [0.6, 0.9]
    mock_gepa.optimize.return_value = mock_result
    mock_gepa.GEPAAdapter = MockGEPAAdapter

    eval_data = sample_data.copy()
    optimizer = _GEPAOptimizer(OptimizerConfig())

    with patch.dict(sys.modules, {"gepa": mock_gepa}):
        optimizer.optimize(
            prompt=sample_prompt,
            target_llm_params=LLMParams(model_name="openai/gpt-4"),
            train_data=sample_data,
            scorers=[sample_scorer],
            eval_data=eval_data,
        )

    call_kwargs = mock_gepa.optimize.call_args.kwargs
    assert call_kwargs["valset"] is not None
    assert len(call_kwargs["valset"]) == len(eval_data)


def test_gepa_optimizer_with_reflection_lm(sample_data, sample_prompt):
    mock_gepa = MagicMock()
    mock_result = Mock()
    mock_result.best_candidate = {"prompt": "Optimized"}
    mock_result.val_aggregate_scores = [0.7, 0.85]
    mock_gepa.optimize.return_value = mock_result
    mock_gepa.GEPAAdapter = MockGEPAAdapter

    config = OptimizerConfig(
        optimizer_llm=LLMParams(model_name="anthropic/claude-3-5-sonnet-20241022")
    )
    optimizer = _GEPAOptimizer(config)

    with patch.dict(sys.modules, {"gepa": mock_gepa}):
        optimizer.optimize(
            prompt=sample_prompt,
            target_llm_params=LLMParams(model_name="openai/gpt-4o"),
            train_data=sample_data,
            scorers=[sample_scorer],
        )

    call_kwargs = mock_gepa.optimize.call_args.kwargs
    assert call_kwargs["reflection_lm"] == "anthropic/claude-3-5-sonnet-20241022"


def test_display_optimization_result_improved(capsys):
    optimizer = _GEPAOptimizer(OptimizerConfig())

    optimizer._display_optimization_result(initial_score=0.5, final_score=0.9)

    captured = capsys.readouterr()
    assert "Initial score: 0.5" in captured.err
    assert "Final score: 0.9" in captured.err
    assert "+0.4000" in captured.err


def test_display_optimization_result_stable(capsys):
    optimizer = _GEPAOptimizer(OptimizerConfig())

    optimizer._display_optimization_result(initial_score=0.8, final_score=0.8)

    captured = capsys.readouterr()
    assert "Score remained stable at: 0.8" in captured.err


def test_display_optimization_result_no_initial(capsys):
    optimizer = _GEPAOptimizer(OptimizerConfig())

    optimizer._display_optimization_result(initial_score=None, final_score=0.9)

    captured = capsys.readouterr()
    assert "Final score: 0.9" in captured.err
    assert "Initial score" not in captured.err


def test_extract_eval_scores():
    optimizer = _GEPAOptimizer(OptimizerConfig())

    mock_result = Mock()
    mock_result.val_aggregate_scores = [0.5, 0.7, 0.9, 0.85]

    initial_score, final_score = optimizer._extract_eval_scores(mock_result)

    assert initial_score == 0.5  # First score
    assert final_score == 0.9  # Max score
