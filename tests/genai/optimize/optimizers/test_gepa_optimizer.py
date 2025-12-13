import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from packaging.version import Version

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
    def eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]):
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="outputs",
                expectations=record["outputs"],
                score=0.8,
                trace={"info": "mock trace"},
                rationales={"score": "mock rationale"},
            )
            for record in dataset
        ]

    return eval_fn


def test_gepa_optimizer_initialization():
    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.max_metric_calls == 100
    assert optimizer.display_progress_bar is False
    assert optimizer.gepa_kwargs == {}


def test_gepa_optimizer_initialization_with_custom_params():
    optimizer = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o",
        max_metric_calls=100,
        display_progress_bar=True,
    )
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.max_metric_calls == 100
    assert optimizer.display_progress_bar is True
    assert optimizer.gepa_kwargs == {}


def test_gepa_optimizer_initialization_with_gepa_kwargs():
    gepa_kwargs_example = {"foo": "bar"}
    optimizer = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o",
        gepa_kwargs=gepa_kwargs_example,
    )
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.max_metric_calls == 100
    assert optimizer.display_progress_bar is False
    assert optimizer.gepa_kwargs == gepa_kwargs_example


def test_gepa_optimizer_optimize(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
):
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
    optimizer = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o-mini", max_metric_calls=50, display_progress_bar=True
    )

    with patch.dict(sys.modules, mock_modules):
        result = optimizer.optimize(
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
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
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

    optimizer = GepaPromptOptimizer(
        reflection_model="anthropic:/claude-3-5-sonnet-20241022",
    )

    with patch.dict(sys.modules, mock_modules):
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["reflection_lm"] == "anthropic/claude-3-5-sonnet-20241022"


def test_gepa_optimizer_optimize_with_custom_gepa_params(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
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

    optimizer = GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o-mini", gepa_kwargs={"foo": "bar"}
    )

    with patch.dict(sys.modules, mock_modules):
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["foo"] == "bar"


def test_gepa_optimizer_optimize_model_name_parsing(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
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

    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["reflection_lm"] == "openai/gpt-4o"


def test_gepa_optimizer_import_error(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
):
    with patch.dict("sys.modules", {"gepa": None}):
        optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

        with pytest.raises(ImportError, match="GEPA is not installed"):
            optimizer.optimize(
                eval_fn=mock_eval_fn,
                train_data=sample_train_data,
                target_prompts=sample_target_prompts,
            )


def test_gepa_optimizer_single_record_dataset(
    sample_target_prompts: dict[str, str], mock_eval_fn: Any
):
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

    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=single_record_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert len(call_kwargs["trainset"]) == 1


def test_gepa_optimizer_custom_adapter_evaluate(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
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

    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        result = optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert "adapter" in call_kwargs
    assert call_kwargs["adapter"] is not None
    assert result.optimized_prompts == sample_target_prompts


def test_make_reflective_dataset_with_traces(
    sample_target_prompts: dict[str, str], mock_eval_fn: Any
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_gepa_module.EvaluationBatch = MagicMock()
    mock_gepa_module.GEPAAdapter = object
    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        captured_adapter = None

        def mock_optimize_fn(**kwargs):
            nonlocal captured_adapter
            captured_adapter = kwargs["adapter"]
            mock_result = Mock()
            mock_result.best_candidate = sample_target_prompts
            mock_result.val_aggregate_scores = []
            return mock_result

        mock_gepa_module.optimize = mock_optimize_fn

        # Call optimize to create the inner adapter
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=[{"inputs": {"question": "test"}, "outputs": "test"}],
            target_prompts=sample_target_prompts,
        )

    # Now test make_reflective_dataset with the captured adapter
    mock_trace = Mock()
    mock_span1 = Mock()
    mock_span1.name = "llm_call"
    mock_span1.inputs = {"prompt": "What is 2+2?"}
    mock_span1.outputs = {"response": "4"}

    mock_span2 = Mock()
    mock_span2.name = "retrieval"
    mock_span2.inputs = {"query": "math"}
    mock_span2.outputs = {"documents": ["doc1", "doc2"]}

    mock_trace.data.spans = [mock_span1, mock_span2]

    # Create mock trajectories
    mock_trajectory1 = Mock()
    mock_trajectory1.trace = mock_trace
    mock_trajectory1.inputs = {"question": "What is 2+2?"}
    mock_trajectory1.outputs = "4"
    mock_trajectory1.expectations = {"expected_response": "4"}

    mock_trajectory2 = Mock()
    mock_trajectory2.trace = None
    mock_trajectory2.inputs = {"question": "What is the capital of France?"}
    mock_trajectory2.outputs = "Paris"
    mock_trajectory2.expectations = {"expected_response": "Paris"}

    # Create mock evaluation batch
    mock_eval_batch = Mock()
    mock_eval_batch.trajectories = [mock_trajectory1, mock_trajectory2]
    mock_eval_batch.scores = [0.9, 0.7]

    # Test make_reflective_dataset
    candidate = {"system_prompt": "You are helpful"}
    components_to_update = ["system_prompt", "instruction"]

    result = captured_adapter.make_reflective_dataset(
        candidate, mock_eval_batch, components_to_update
    )

    # Verify result structure
    assert isinstance(result, dict)
    assert "system_prompt" in result
    assert "instruction" in result

    system_data = result["system_prompt"]
    assert len(system_data) == 2
    assert system_data[0]["component_name"] == "system_prompt"
    assert system_data[0]["current_text"] == "You are helpful"
    assert system_data[0]["score"] == 0.9
    assert system_data[0]["inputs"] == {"question": "What is 2+2?"}
    assert system_data[0]["outputs"] == "4"
    assert system_data[0]["expectations"] == {"expected_response": "4"}
    assert system_data[0]["index"] == 0

    # Verify trace spans
    assert len(system_data[0]["trace"]) == 2
    assert system_data[0]["trace"][0]["name"] == "llm_call"
    assert system_data[0]["trace"][0]["inputs"] == {"prompt": "What is 2+2?"}
    assert system_data[0]["trace"][0]["outputs"] == {"response": "4"}
    assert system_data[0]["trace"][1]["name"] == "retrieval"

    # Verify second record (no trace)
    assert system_data[1]["trace"] == []
    assert system_data[1]["score"] == 0.7
    assert system_data[1]["inputs"] == {"question": "What is the capital of France?"}
    assert system_data[1]["outputs"] == "Paris"
    assert system_data[1]["expectations"] == {"expected_response": "Paris"}


@pytest.mark.parametrize("gepa_version", ["0.0.9", "0.0.18", "0.1.0"])
@pytest.mark.parametrize("enable_tracking", [True, False])
def test_gepa_optimizer_version_check(
    sample_train_data: list[dict[str, Any]],
    sample_target_prompts: dict[str, str],
    mock_eval_fn: Any,
    gepa_version: str,
    enable_tracking: bool,
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

    optimizer = GepaPromptOptimizer(reflection_model="openai:/gpt-4o")

    with (
        patch.dict(sys.modules, mock_modules),
        patch("importlib.metadata.version", return_value=gepa_version),
    ):
        optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
            enable_tracking=enable_tracking,
        )

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs

    if Version(gepa_version) < Version("0.0.10"):
        assert "use_mlflow" not in call_kwargs
    else:
        assert "use_mlflow" in call_kwargs
        assert call_kwargs["use_mlflow"] == enable_tracking
