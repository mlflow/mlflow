import json
import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.metaprompt_optimizer import MetaPromptOptimizer
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
        "instruction": "Answer the following question: {{question}}",
    }


@pytest.fixture
def sample_target_prompts_multiple():
    return {
        "system_prompt": "You are a helpful assistant.",
        "instruction": "Answer the following question: {{question}}",
    }


@pytest.fixture
def mock_eval_fn():
    """Mock evaluation function that returns varied scores."""

    def eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]):
        # Return varied scores for diverse sampling
        scores = [0.9, 0.7, 0.4, 0.2]  # High to low
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="mock output",
                expectations=record["outputs"],
                score=scores[i % len(scores)],
                trace=Mock(),  # Use Mock for trace
                rationales={"correctness": f"Score {scores[i % len(scores)]}"},
            )
            for i, record in enumerate(dataset)
        ]

    return eval_fn


@pytest.fixture
def mock_litellm_response():
    """Mock litellm response with improved prompts."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {"instruction": "Improved: Answer this question carefully: {{question}}"}
    )
    return mock_response


# Initialization tests


def test_metaprompt_optimizer_initialization():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.num_iterations == 3
    assert optimizer.num_examples is None
    assert optimizer.lm_kwargs == {}
    assert optimizer.display_progress_bar is False


def test_metaprompt_optimizer_initialization_with_custom_params():
    optimizer = MetaPromptOptimizer(
        reflection_model="anthropic:/claude-3-5-sonnet-20241022",
        num_iterations=5,
        num_examples=20,
        lm_kwargs={"temperature": 0.9, "max_tokens": 4096},
        display_progress_bar=True,
    )
    assert optimizer.reflection_model == "anthropic:/claude-3-5-sonnet-20241022"
    assert optimizer.num_iterations == 5
    assert optimizer.num_examples == 20
    assert optimizer.lm_kwargs == {"temperature": 0.9, "max_tokens": 4096}
    assert optimizer.display_progress_bar is True


def test_metaprompt_optimizer_invalid_num_iterations():
    with pytest.raises(MlflowException, match="`num_iterations` must be at least 1"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=0)


def test_metaprompt_optimizer_invalid_num_examples():
    with pytest.raises(MlflowException, match="`num_examples` must be at least 1 or None"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_examples=0)


def test_metaprompt_optimizer_invalid_lm_kwargs():
    with pytest.raises(MlflowException, match="`lm_kwargs` must be a dictionary"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", lm_kwargs="invalid")

    with pytest.raises(MlflowException, match="`lm_kwargs` must be a dictionary"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", lm_kwargs=123)


# Template variable tests


def test_extract_template_variables():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    prompts = {
        "instruction": "Answer {{question}} about {{topic}}",
        "system": "You are a {{role}}",
    }

    variables = optimizer._extract_template_variables(prompts)

    assert variables["instruction"] == {"question", "topic"}
    assert variables["system"] == {"role"}


def test_extract_template_variables_no_vars():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    prompts = {"instruction": "Answer the question"}

    variables = optimizer._extract_template_variables(prompts)

    assert variables["instruction"] == set()


def test_validate_template_variables_success():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}"}
    new = {"instruction": "Please answer this {{question}} carefully"}

    # Should not raise
    assert optimizer._validate_template_variables(original, new) is True


def test_validate_template_variables_missing_var():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}"}
    new = {"instruction": "Answer the question"}  # Missing {{question}}

    with pytest.raises(MlflowException, match="Missing.*question"):
        optimizer._validate_template_variables(original, new)


def test_validate_template_variables_extra_var():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}"}
    new = {"instruction": "Answer {{question}} about {{topic}}"}  # Extra {{topic}}

    with pytest.raises(MlflowException, match="Extra.*topic"):
        optimizer._validate_template_variables(original, new)


def test_validate_template_variables_missing_prompt():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}", "system": "You are helpful"}
    new = {"instruction": "Answer {{question}}"}  # Missing system prompt

    with pytest.raises(MlflowException, match="Prompt 'system' missing"):
        optimizer._validate_template_variables(original, new)


# Example sampling tests


def test_sample_examples_with_num_examples(sample_train_data, mock_eval_fn):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_examples=2)
    eval_results = mock_eval_fn({}, sample_train_data)

    sampled = optimizer._sample_examples(sample_train_data, eval_results)

    assert len(sampled) == 2
    # Verify samples are from the original data
    assert all(s in list(zip(sample_train_data, eval_results)) for s in sampled)


def test_sample_examples_with_none(sample_train_data, mock_eval_fn):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_examples=None)
    eval_results = mock_eval_fn({}, sample_train_data)

    # Should return all examples when num_examples is None
    sampled = optimizer._sample_examples(sample_train_data, eval_results)

    assert len(sampled) == len(sample_train_data)


def test_sample_examples_more_than_available(sample_train_data, mock_eval_fn):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_examples=10)
    eval_results = mock_eval_fn({}, sample_train_data)

    # Request more examples than available
    sampled = optimizer._sample_examples(sample_train_data, eval_results)

    # Should return all available
    assert len(sampled) == len(sample_train_data)


# Meta-prompt building tests


def test_build_zero_shot_meta_prompt(sample_target_prompts):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    template_vars = optimizer._extract_template_variables(sample_target_prompts)

    meta_prompt = optimizer._build_zero_shot_meta_prompt(sample_target_prompts, template_vars)

    assert "PROMPT ENGINEERING BEST PRACTICES" in meta_prompt
    assert "{{question}}" in meta_prompt or "question" in meta_prompt
    assert "instruction" in meta_prompt
    assert "JSON" in meta_prompt


def test_build_few_shot_meta_prompt(sample_train_data, sample_target_prompts, mock_eval_fn):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_examples=2)
    template_vars = optimizer._extract_template_variables(sample_target_prompts)
    eval_results = mock_eval_fn(sample_target_prompts, sample_train_data)

    sampled = optimizer._sample_examples(sample_train_data, eval_results)

    meta_prompt = optimizer._build_few_shot_meta_prompt(sample_target_prompts, template_vars, sampled)

    assert "EVALUATION EXAMPLES" in meta_prompt
    assert "Example 1:" in meta_prompt
    assert "Score:" in meta_prompt
    assert "JSON" in meta_prompt


def test_format_examples(sample_train_data, mock_eval_fn):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    eval_results = mock_eval_fn({}, sample_train_data)
    examples = list(zip(sample_train_data[:2], eval_results[:2]))

    formatted = optimizer._format_examples(examples)

    assert "Example 1:" in formatted
    assert "Example 2:" in formatted
    assert "Input:" in formatted
    assert "Output:" in formatted
    assert "Score:" in formatted


# LLM invocation tests


def test_call_reflection_model_success(mock_litellm_response):
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
        result = optimizer._call_reflection_model("test prompt")

        assert isinstance(result, dict)
        assert "instruction" in result
        assert "{{question}}" in result["instruction"]

        # Verify litellm.completion was called with correct base parameters
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["max_retries"] == 3


def test_call_reflection_model_with_markdown():
    # Test response with markdown code blocks
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = """```json
{
  "instruction": "Improved: Answer {{question}}"
}
```"""

    with patch("litellm.completion", return_value=mock_response):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
        result = optimizer._call_reflection_model("test prompt")

        assert isinstance(result, dict)
        assert "instruction" in result


def test_call_reflection_model_litellm_not_installed():
    with patch.dict(sys.modules, {"litellm": None}):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        with pytest.raises(ImportError, match="litellm is required"):
            optimizer._call_reflection_model("test prompt")


def test_call_reflection_model_llm_failure():
    with patch("litellm.completion", side_effect=Exception("API error")):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        with pytest.raises(MlflowException, match="Failed to call reflection model"):
            optimizer._call_reflection_model("test prompt")


def test_call_reflection_model_with_lm_kwargs(mock_litellm_response):
    """Test that lm_kwargs are properly passed to litellm.completion."""
    custom_lm_kwargs = {"temperature": 0.5, "max_tokens": 2048, "top_p": 0.9}

    with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
        optimizer = MetaPromptOptimizer(
            reflection_model="openai:/gpt-4o", lm_kwargs=custom_lm_kwargs
        )
        result = optimizer._call_reflection_model("test prompt")

        assert isinstance(result, dict)

        # Verify that custom lm_kwargs were passed through
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["top_p"] == 0.9
        # Also verify base parameters are still present
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["max_retries"] == 3


# Response parsing tests


# Score computation tests


def test_compute_aggregate_score():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    results = [
        Mock(score=0.8),
        Mock(score=0.6),
        Mock(score=0.9),
    ]

    score = optimizer._compute_aggregate_score(results)

    assert score == pytest.approx(0.7667, rel=0.01)


def test_compute_aggregate_score_empty():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    results = []

    score = optimizer._compute_aggregate_score(results)

    assert score == 0.0


# Iteration history formatting tests


# Integration tests - Zero-shot mode


def test_optimize_zero_shot_mode(sample_target_prompts, mock_litellm_response):
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=5)

            result = optimizer.optimize(
                eval_fn=Mock(),  # Not used in zero-shot
                train_data=[],  # Empty triggers zero-shot
                target_prompts=sample_target_prompts,
                enable_tracking=False,
            )

            assert isinstance(result, PromptOptimizerOutput)
            assert result.initial_eval_score is None  # No evaluation in zero-shot
            assert result.final_eval_score is None
            assert "instruction" in result.optimized_prompts
            assert "{{question}}" in result.optimized_prompts["instruction"]
            # Zero-shot always uses single iteration regardless of num_iterations setting
            assert mock_completion.call_count == 1


# Integration tests - Few-shot mode


def test_optimize_few_shot_mode(
    sample_train_data,
    sample_target_prompts,
    mock_eval_fn,
    mock_litellm_response,
):
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
            optimizer = MetaPromptOptimizer(
                reflection_model="openai:/gpt-4o", num_iterations=2, num_examples=2
            )

            result = optimizer.optimize(
                eval_fn=mock_eval_fn,
                train_data=sample_train_data,
                target_prompts=sample_target_prompts,
                enable_tracking=False,
            )

            assert isinstance(result, PromptOptimizerOutput)
            assert result.initial_eval_score is not None
            assert result.final_eval_score is not None
            assert "instruction" in result.optimized_prompts
            assert mock_completion.call_count == 2  # num_iterations


def test_optimize_few_shot_with_improvement(sample_train_data, sample_target_prompts):
    """Test that optimizer tracks improvement when scores increase."""
    # Mock litellm to return improved prompts
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {"instruction": "Better: Answer {{question}}"}
    )

    # Mock eval_fn that returns improving scores
    call_count = [0]

    def improving_eval_fn(candidate_prompts, dataset):
        call_count[0] += 1
        score = 0.5 + (call_count[0] * 0.1)  # Increasing score each call
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="mock output",
                expectations=record["outputs"],
                score=score,
                trace=Mock(),
                rationales={},
            )
            for record in dataset
        ]

    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_response):
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=3)

            result = optimizer.optimize(
                eval_fn=improving_eval_fn,
                train_data=sample_train_data,
                target_prompts=sample_target_prompts,
                enable_tracking=False,
            )

            # Should see improvement
            assert result.final_eval_score > result.initial_eval_score


def test_optimize_preserves_template_variables(sample_train_data, mock_eval_fn):
    """Test that optimization fails if LLM drops template variables."""
    # Mock response that drops the {{question}} variable
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {"instruction": "Answer the question"}  # Missing {{question}}
    )

    prompts = {"instruction": "Answer {{question}}"}

    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_response):
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=1)

            result = optimizer.optimize(
                eval_fn=mock_eval_fn,
                train_data=sample_train_data,
                target_prompts=prompts,
                enable_tracking=False,
            )

            # Should keep original prompts due to validation failure
            # (caught as exception and logged as warning)
            assert "{{question}}" in result.optimized_prompts["instruction"]


def test_optimize_with_multiple_prompts(
    sample_train_data,
    sample_target_prompts_multiple,
    mock_eval_fn,
):
    """Test optimization with multiple prompts."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "system_prompt": "Improved: You are an expert assistant.",
            "instruction": "Improved: Answer {{question}}",
        }
    )

    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_response):
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=1)

            result = optimizer.optimize(
                eval_fn=mock_eval_fn,
                train_data=sample_train_data,
                target_prompts=sample_target_prompts_multiple,
                enable_tracking=False,
            )

            assert "system_prompt" in result.optimized_prompts
            assert "instruction" in result.optimized_prompts
            assert "{{question}}" in result.optimized_prompts["instruction"]


def test_optimize_with_tracking_enabled(
    sample_train_data, sample_target_prompts, mock_eval_fn, mock_litellm_response
):
    """Test that iteration metrics are logged when tracking is enabled."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response):
            with patch("mlflow.log_metric") as mock_log_metric:
                optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=2)

                result = optimizer.optimize(
                    eval_fn=mock_eval_fn,
                    train_data=sample_train_data,
                    target_prompts=sample_target_prompts,
                    enable_tracking=True,
                )

                # Verify optimization completed successfully
                assert isinstance(result, PromptOptimizerOutput)
                assert result.initial_eval_score is not None
                assert result.final_eval_score is not None

                # Verify iteration metrics were logged
                # Should log iteration_1_score, iteration_1_improvement, iteration_2_score, iteration_2_improvement
                assert mock_log_metric.call_count >= 2  # At least score metrics for each iteration

                # Check that iteration scores were logged
                logged_metrics = [call[0][0] for call in mock_log_metric.call_args_list]
                assert "iteration_1_score" in logged_metrics
                assert "iteration_2_score" in logged_metrics


def test_optimize_with_tracking_disabled(
    sample_train_data, sample_target_prompts, mock_eval_fn, mock_litellm_response
):
    """Test that no iteration metrics are logged when tracking is disabled."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response):
            with patch("mlflow.log_metric") as mock_log_metric:
                optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=2)

                result = optimizer.optimize(
                    eval_fn=mock_eval_fn,
                    train_data=sample_train_data,
                    target_prompts=sample_target_prompts,
                    enable_tracking=False,
                )

                # Verify optimization completed successfully
                assert isinstance(result, PromptOptimizerOutput)
                assert result.initial_eval_score is not None
                assert result.final_eval_score is not None

                # Verify NO iteration metrics were logged
                mock_log_metric.assert_not_called()


def test_optimize_zero_shot_with_guidelines(sample_target_prompts, mock_litellm_response):
    """Test that guidelines are included in zero-shot meta-prompt."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

            custom_guidelines = (
                "This is for a finance advisor to project tax situations. "
                "Do not include information outside of finance."
            )

            result = optimizer.optimize(
                eval_fn=Mock(),
                train_data=[],  # Empty triggers zero-shot
                target_prompts=sample_target_prompts,
                guidelines=custom_guidelines,
            )

            # Verify optimization completed
            assert isinstance(result, PromptOptimizerOutput)
            assert result.optimized_prompts is not None

            # Verify guidelines were included in the meta-prompt
            mock_completion.assert_called_once()
            meta_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
            assert "CUSTOM GUIDELINES:" in meta_prompt
            assert custom_guidelines in meta_prompt


def test_optimize_few_shot_with_guidelines(
    sample_train_data, sample_target_prompts, mock_eval_fn, mock_litellm_response
):
    """Test that guidelines are included in few-shot meta-prompt."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
            with patch("mlflow.log_metric"):  # Patch to avoid run leakage
                optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", num_iterations=1)

                custom_guidelines = (
                    "This is for a finance advisor to project tax situations. "
                    "Do not include information outside of finance."
                )

                result = optimizer.optimize(
                    eval_fn=mock_eval_fn,
                    train_data=sample_train_data,
                    target_prompts=sample_target_prompts,
                    guidelines=custom_guidelines,
                    enable_tracking=False,  # Disable tracking to avoid run issues
                )

                # Verify optimization completed
                assert isinstance(result, PromptOptimizerOutput)
                assert result.initial_eval_score is not None
                assert result.final_eval_score is not None

                # Verify guidelines were included in the meta-prompt (should be called twice: baseline + iteration)
                # The iteration call is the second one
                assert mock_completion.call_count == 1  # One iteration
                meta_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
                assert "CUSTOM GUIDELINES:" in meta_prompt
                assert custom_guidelines in meta_prompt


def test_optimize_zero_shot_without_guidelines(sample_target_prompts, mock_litellm_response):
    """Test that meta-prompt works correctly when no guidelines are provided."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
            optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

            result = optimizer.optimize(
                eval_fn=Mock(),
                train_data=[],  # Empty triggers zero-shot
                target_prompts=sample_target_prompts,
                guidelines=None,
            )

            # Verify optimization completed
            assert isinstance(result, PromptOptimizerOutput)
            assert result.optimized_prompts is not None

            # Verify no "CUSTOM GUIDELINES:" section in meta-prompt
            mock_completion.assert_called_once()
            meta_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
            assert "CUSTOM GUIDELINES:" not in meta_prompt


def test_build_zero_shot_meta_prompt_with_guidelines(sample_target_prompts):
    """Test that _build_zero_shot_meta_prompt includes guidelines."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
        template_vars = optimizer._extract_template_variables(sample_target_prompts)

        custom_guidelines = "Focus on concise, accurate answers for finance domain."

        meta_prompt = optimizer._build_zero_shot_meta_prompt(
            sample_target_prompts, template_vars, custom_guidelines
        )

        # Verify structure
        assert "CUSTOM GUIDELINES:" in meta_prompt
        assert custom_guidelines in meta_prompt
        assert "TEMPLATE VARIABLES:" in meta_prompt
        assert "PROMPT ENGINEERING BEST PRACTICES:" in meta_prompt


def test_build_few_shot_meta_prompt_with_guidelines(sample_target_prompts):
    """Test that _build_few_shot_meta_prompt includes guidelines."""
    with patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
        template_vars = optimizer._extract_template_variables(sample_target_prompts)

        # Create sample examples
        sampled_examples = [
            (
                {"inputs": {"question": "test"}, "outputs": "answer"},
                EvaluationResultRecord(
                    inputs={"question": "test"},
                    outputs="answer",
                    expectations="answer",
                    score=0.8,
                    trace=Mock(),
                    rationales={"correctness": "Good"},
                ),
            )
        ]

        custom_guidelines = "Focus on concise, accurate answers for finance domain."

        meta_prompt = optimizer._build_few_shot_meta_prompt(
            sample_target_prompts, template_vars, sampled_examples, custom_guidelines
        )

        # Verify structure
        assert "CUSTOM GUIDELINES:" in meta_prompt
        assert custom_guidelines in meta_prompt
        assert "TEMPLATE VARIABLES:" in meta_prompt
        assert "EVALUATION EXAMPLES:" in meta_prompt
        assert "CURRENT PERFORMANCE:" in meta_prompt


def test_optimize_with_val_data(sample_train_data, sample_target_prompts, mock_litellm_response):
    """Test that val_data is used for evaluation while train_data is used for meta-prompting."""
    # Create separate train and validation datasets
    train_data = sample_train_data[:2]  # First 2 examples for training
    val_data = sample_train_data[2:]  # Last 2 examples for validation

    # Track which dataset was used for each eval_fn call
    eval_calls = []

    def tracking_eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]):
        eval_calls.append({"prompts": candidate_prompts, "dataset": dataset})
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="outputs",
                expectations=record["outputs"],
                score=0.8 if dataset == val_data else 0.7,  # Higher score on val_data
                trace=Mock(),
                rationales={"score": "mock rationale"},
            )
            for record in dataset
        ]

    with (
        patch("litellm.completion", return_value=mock_litellm_response),
        patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")),
        patch("mlflow.log_metric"),
    ):
        optimizer = MetaPromptOptimizer(
            reflection_model="openai:/gpt-4o", num_iterations=2, num_examples=None
        )
        result = optimizer.optimize(
            eval_fn=tracking_eval_fn,
            train_data=train_data,
            target_prompts=sample_target_prompts,
            enable_tracking=False,
            val_data=val_data,
        )

    # Verify that eval_fn was called with val_data (not train_data) for evaluation
    # Should have: 1 baseline + 2 iterations = 3 calls
    assert len(eval_calls) == 3

    # All evaluation calls should use val_data
    for call in eval_calls:
        assert call["dataset"] == val_data, "eval_fn should be called with val_data, not train_data"

    # Verify result scores come from val_data evaluation
    assert isinstance(result, PromptOptimizerOutput)
    assert result.initial_eval_score == 0.8
    assert result.final_eval_score == 0.8


def test_optimize_without_val_data_uses_train_data(
    sample_train_data, sample_target_prompts, mock_litellm_response
):
    """Test that when val_data is not provided, train_data is used for evaluation."""
    eval_calls = []

    def tracking_eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]):
        eval_calls.append({"prompts": candidate_prompts, "dataset": dataset})
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="outputs",
                expectations=record["outputs"],
                score=0.8,
                trace=Mock(),
                rationales={"score": "mock rationale"},
            )
            for record in dataset
        ]

    with (
        patch("litellm.completion", return_value=mock_litellm_response),
        patch("mlflow.metrics.genai.model_utils._parse_model_uri", return_value=("openai", "gpt-4o")),
        patch("mlflow.log_metric"),
    ):
        optimizer = MetaPromptOptimizer(
            reflection_model="openai:/gpt-4o", num_iterations=1, num_examples=None
        )
        result = optimizer.optimize(
            eval_fn=tracking_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
            enable_tracking=False,
            val_data=None,  # Explicitly no val_data
        )

    # Verify that eval_fn was called with train_data
    # Should have: 1 baseline + 1 iteration = 2 calls
    assert len(eval_calls) == 2

    # All evaluation calls should use train_data when val_data is None
    for call in eval_calls:
        assert (
            call["dataset"] == sample_train_data
        ), "eval_fn should be called with train_data when val_data is None"

    assert isinstance(result, PromptOptimizerOutput)
