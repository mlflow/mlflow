import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.genai.optimize.optimizers.gepa_optimizer import GepaPromptOptimizer
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput


@pytest.fixture
def sample_train_data():
    return [
        {
            "inputs": {"question": "What is 2+2?"},
            "outputs": "4",
        },
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": "Paris",
        },
        {
            "inputs": {"question": "What is 3*3?"},
            "outputs": "9",
        },
        {
            "inputs": {"question": "What color is the sky?"},
            "outputs": "Blue",
        },
    ]


@pytest.fixture
def sample_target_prompts():
    return {
        "system_prompt": "You are a helpful assistant.",
        "instruction": "Answer the following question: {{question}}",
    }


@pytest.fixture
def mock_eval_fn():
    def eval_fn(candidate_prompts, dataset):
        results = []
        for record in dataset:
            results.append(
                EvaluationResultRecord(
                    inputs=record["inputs"],
                    outputs="outputs",
                    expectations=record["outputs"],
                    score=0.8,
                    trace={"info": "mock trace"},
                )
            )
        return results

    return eval_fn


def test_gepa_optimizer_initialization():
    adapter = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")
    assert adapter.reflection_model == "openai:/gpt-4o"
    assert adapter.max_metric_calls == 100
    assert adapter.display_progress_bar is False


def test_gepa_optimizer_initialization_with_custom_params():
    adapter = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o",
        max_metric_calls=100,
        display_progress_bar=True,
    )
    assert adapter.reflection_model == "openai:/gpt-4o"
    assert adapter.max_metric_calls == 100
    assert adapter.display_progress_bar is True


def test_gepa_optimizer_optimize(sample_train_data, sample_target_prompts, mock_eval_fn):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_result = Mock()
    mock_result.best_candidate = {
        "system_prompt": "You are a highly skilled assistant.",
        "instruction": "Please answer this question carefully: {{question}}",
    }
    mock_result.val_aggregate_scores = [0.5, 0.6, 0.8, 0.9]  # Mock scores for testing
    mock_gepa_module.optimize.return_value = mock_result
    mock_gepa_module.EvaluationBatch = MagicMock()
    adapter = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o-mini", max_metric_calls=50, display_progress_bar=True
    )

    with patch.dict(sys.modules, mock_modules):
        result = adapter.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    # Verify result
    assert isinstance(result, PromptOptimizerOutput)
    assert result.optimized_prompts == mock_result.best_candidate
    assert "system_prompt" in result.optimized_prompts
    assert "instruction" in result.optimized_prompts
    # Verify scores are extracted
    assert result.initial_eval_score == 0.5  # First score
    assert result.final_eval_score == 0.9  # Max score

    # Verify GEPA was called with correct parameters
    mock_gepa_module.optimize.assert_called_once()
    call_kwargs = mock_gepa_module.optimize.call_args.kwargs

    assert call_kwargs["seed_candidate"] == sample_target_prompts
    assert call_kwargs["adapter"] is not None
    assert call_kwargs["max_metric_calls"] == 50
    assert call_kwargs["reflection_lm"] == "openai/gpt-4o-mini"
    assert call_kwargs["display_progress_bar"] is True
    assert len(call_kwargs["trainset"]) == 4


def test_gepa_optimizer_optimize_with_custom_reflection_model(
    sample_train_data, sample_target_prompts, mock_eval_fn
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_result = Mock()
    mock_result.best_candidate = sample_target_prompts
    mock_result.val_aggregate_scores = []
    mock_gepa_module.optimize.return_value = mock_result
    mock_gepa_module.EvaluationBatch = MagicMock()

    adapter = GepaPromptOptimizer(
        reflection_model="anthropic:/claude-3-5-sonnet-20241022",
    )

    with patch.dict(sys.modules, mock_modules):
        adapter.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["reflection_lm"] == "anthropic/claude-3-5-sonnet-20241022"


def test_gepa_optimizer_optimize_model_name_parsing(
    sample_train_data, sample_target_prompts, mock_eval_fn
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_result = Mock()
    mock_result.best_candidate = sample_target_prompts
    mock_result.val_aggregate_scores = []
    mock_gepa_module.optimize.return_value = mock_result
    mock_gepa_module.EvaluationBatch = MagicMock()

    adapter = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        adapter.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["reflection_lm"] == "openai/gpt-4o"


def test_gepa_optimizer_import_error(sample_train_data, sample_target_prompts, mock_eval_fn):
    with patch.dict("sys.modules", {"gepa": None}):
        adapter = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

        with pytest.raises(ImportError, match="GEPA is not installed"):
            adapter.optimize(
                eval_fn=mock_eval_fn,
                train_data=sample_train_data,
                target_prompts=sample_target_prompts,
            )


def test_gepa_optimizer_single_record_dataset(sample_target_prompts, mock_eval_fn):
    single_record_data = [
        {
            "inputs": {"question": "What is 2+2?"},
            "outputs": "4",
        }
    ]

    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_result = Mock()
    mock_result.best_candidate = sample_target_prompts
    mock_result.val_aggregate_scores = []
    mock_gepa_module.optimize.return_value = mock_result
    mock_gepa_module.EvaluationBatch = MagicMock()

    adapter = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        adapter.optimize(
            eval_fn=mock_eval_fn,
            train_data=single_record_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert len(call_kwargs["trainset"]) == 1


def test_gepa_optimizer_custom_adapter_evaluate(
    sample_train_data, sample_target_prompts, mock_eval_fn
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_result = Mock()
    mock_result.best_candidate = sample_target_prompts
    mock_result.val_aggregate_scores = []
    mock_gepa_module.optimize.return_value = mock_result
    mock_gepa_module.EvaluationBatch = MagicMock()

    adapter = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        result = adapter.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert "adapter" in call_kwargs
    assert call_kwargs["adapter"] is not None
    assert result.optimized_prompts == sample_target_prompts
