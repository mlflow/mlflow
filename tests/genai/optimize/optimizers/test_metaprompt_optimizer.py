import json
import sys
from typing import Any
from unittest.mock import Mock, patch

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


def mock_eval_fn(candidate_prompts: dict[str, str], dataset: list[dict[str, Any]]):
    """Mock evaluation function that returns varied scores."""
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


def test_metaprompt_optimizer_initialization():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    assert optimizer.reflection_model == "openai:/gpt-4o"
    assert optimizer.lm_kwargs == {}


def test_metaprompt_optimizer_initialization_with_custom_params():
    optimizer = MetaPromptOptimizer(
        reflection_model="anthropic:/claude-3-5-sonnet-20241022",
        lm_kwargs={"temperature": 0.9, "max_tokens": 4096},
    )
    assert optimizer.reflection_model == "anthropic:/claude-3-5-sonnet-20241022"
    assert optimizer.lm_kwargs == {"temperature": 0.9, "max_tokens": 4096}


def test_metaprompt_optimizer_invalid_lm_kwargs():
    with pytest.raises(MlflowException, match="`lm_kwargs` must be a dictionary"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", lm_kwargs="invalid")

    with pytest.raises(MlflowException, match="`lm_kwargs` must be a dictionary"):
        MetaPromptOptimizer(reflection_model="openai:/gpt-4o", lm_kwargs=123)


def test_extract_template_variables():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    prompts = {
        "instruction": "Answer {{question}} about {{topic}}",
        "system": "You are a {{role}}",
    }

    variables = optimizer._extract_template_variables(prompts)

    assert variables["instruction"] == {"question", "topic"}
    assert variables["system"] == {"role"}


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


def test_validate_prompt_names_missing():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}", "system": "You are helpful"}
    new = {"instruction": "Answer {{question}}"}

    with pytest.raises(MlflowException, match="Prompts missing.*system"):
        optimizer._validate_prompt_names(original, new)


def test_validate_prompt_names_unexpected():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}"}
    new = {
        "instruction": "Answer {{question}}",
        "extra_prompt": "This is unexpected",
    }

    with pytest.raises(MlflowException, match="Unexpected prompts.*extra_prompt"):
        optimizer._validate_prompt_names(original, new)


def test_validate_prompt_names_success():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    original = {"instruction": "Answer {{question}}", "system": "You are helpful"}
    new = {"instruction": "Answer {{question}}", "system": "You are an expert"}

    assert optimizer._validate_prompt_names(original, new) is True


def test_build_zero_shot_meta_prompt(sample_target_prompts):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    template_vars = optimizer._extract_template_variables(sample_target_prompts)

    meta_prompt = optimizer._build_zero_shot_meta_prompt(sample_target_prompts, template_vars)

    assert "PROMPT ENGINEERING BEST PRACTICES" in meta_prompt
    assert "{{question}}" in meta_prompt or "question" in meta_prompt
    assert "instruction" in meta_prompt
    assert "JSON" in meta_prompt


def test_build_few_shot_meta_prompt(sample_train_data, sample_target_prompts):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    template_vars = optimizer._extract_template_variables(sample_target_prompts)
    eval_results = mock_eval_fn(sample_target_prompts, sample_train_data)

    meta_prompt = optimizer._build_few_shot_meta_prompt(
        sample_target_prompts, template_vars, eval_results
    )

    assert "EVALUATION EXAMPLES" in meta_prompt
    assert "Example 1:" in meta_prompt
    assert "Score:" in meta_prompt
    assert "JSON" in meta_prompt


def test_build_few_shot_meta_prompt_empty_eval_results(sample_target_prompts):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    template_vars = optimizer._extract_template_variables(sample_target_prompts)

    with pytest.raises(MlflowException, match="Few-shot metaprompting requires evaluation results"):
        optimizer._build_few_shot_meta_prompt(sample_target_prompts, template_vars, [])


def test_format_examples(sample_train_data):
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    eval_results = mock_eval_fn({}, sample_train_data)

    formatted = optimizer._format_examples(eval_results[:2])

    assert "Example 1:" in formatted
    assert "Example 2:" in formatted
    assert "Input:" in formatted
    assert "Output:" in formatted
    assert "Score:" in formatted


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


def test_optimize_zero_shot_mode(sample_target_prompts, mock_litellm_response):
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

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
        # Zero-shot uses single pass
        assert mock_completion.call_count == 1


def test_optimize_few_shot_mode(sample_train_data, sample_target_prompts, mock_litellm_response):
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_completion:
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        result = optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
            enable_tracking=False,
        )

        assert isinstance(result, PromptOptimizerOutput)
        assert result.initial_eval_score is not None
        assert result.final_eval_score is not None  # Sanity check evaluation on train data
        assert "instruction" in result.optimized_prompts
        assert mock_completion.call_count == 1  # Single pass


def test_optimize_few_shot_with_baseline_eval(sample_train_data, sample_target_prompts):
    # Mock litellm to return improved prompts
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {"instruction": "Better: Answer {{question}}"}
    )

    # Mock eval_fn that returns scores
    def mock_eval_fn(candidate_prompts, dataset):
        return [
            EvaluationResultRecord(
                inputs=record["inputs"],
                outputs="mock output",
                expectations=record["outputs"],
                score=0.7,
                trace=Mock(),
                rationales={},
            )
            for record in dataset
        ]

    with patch("litellm.completion", return_value=mock_response):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        result = optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts,
            enable_tracking=False,
        )

        # Should have both baseline and final eval scores (sanity check)
        assert result.initial_eval_score is not None
        assert result.final_eval_score is not None
        assert "Better" in result.optimized_prompts["instruction"]


def test_optimize_preserves_template_variables(sample_train_data):
    # Mock response that drops the {{question}} variable
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {"instruction": "Answer the question"}  # Missing {{question}}
    )

    prompts = {"instruction": "Answer {{question}}"}

    with patch("litellm.completion", return_value=mock_response):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        result = optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=prompts,
            enable_tracking=False,
        )

        # Should keep original prompts due to validation failure
        # (caught as exception and logged as warning)
        assert "{{question}}" in result.optimized_prompts["instruction"]


def test_optimize_with_multiple_prompts(sample_train_data, sample_target_prompts_multiple):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "system_prompt": "Improved: You are an expert assistant.",
            "instruction": "Improved: Answer {{question}}",
        }
    )

    with patch("litellm.completion", return_value=mock_response):
        optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

        result = optimizer.optimize(
            eval_fn=mock_eval_fn,
            train_data=sample_train_data,
            target_prompts=sample_target_prompts_multiple,
            enable_tracking=False,
        )

        assert "system_prompt" in result.optimized_prompts
        assert "instruction" in result.optimized_prompts
        assert "{{question}}" in result.optimized_prompts["instruction"]


def test_build_zero_shot_meta_prompt_with_guidelines(sample_target_prompts):
    custom_guidelines = "Focus on concise, accurate answers for finance domain."
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", guidelines=custom_guidelines)
    template_vars = optimizer._extract_template_variables(sample_target_prompts)

    meta_prompt = optimizer._build_zero_shot_meta_prompt(sample_target_prompts, template_vars)

    # Verify structure
    assert "CUSTOM GUIDELINES:" in meta_prompt
    assert custom_guidelines in meta_prompt
    assert "TEMPLATE VARIABLES:" in meta_prompt
    assert "PROMPT ENGINEERING BEST PRACTICES:" in meta_prompt


def test_build_few_shot_meta_prompt_with_guidelines(sample_target_prompts):
    custom_guidelines = "Focus on concise, accurate answers for finance domain."
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o", guidelines=custom_guidelines)
    template_vars = optimizer._extract_template_variables(sample_target_prompts)

    # Create sample evaluation results
    eval_results = [
        EvaluationResultRecord(
            inputs={"question": "test"},
            outputs="answer",
            expectations="answer",
            score=0.8,
            trace=Mock(),
            rationales={"correctness": "Good"},
        )
    ]

    meta_prompt = optimizer._build_few_shot_meta_prompt(
        sample_target_prompts, template_vars, eval_results
    )

    # Verify structure
    assert "CUSTOM GUIDELINES:" in meta_prompt
    assert custom_guidelines in meta_prompt
    assert "TEMPLATE VARIABLES:" in meta_prompt
    assert "EVALUATION EXAMPLES" in meta_prompt  # Now includes score in header
    assert "Current Score:" in meta_prompt


def test_compute_per_scorer_scores():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

    # Test with multiple results having individual scores
    eval_results = [
        EvaluationResultRecord(
            inputs={"q": "1"},
            outputs="a",
            expectations="a",
            score=0.8,
            trace=Mock(),
            rationales={},
            individual_scores={"Correctness": 0.9, "Safety": 0.7},
        ),
        EvaluationResultRecord(
            inputs={"q": "2"},
            outputs="b",
            expectations="b",
            score=0.6,
            trace=Mock(),
            rationales={},
            individual_scores={"Correctness": 0.7, "Safety": 0.5},
        ),
    ]

    per_scorer = optimizer._compute_per_scorer_scores(eval_results)

    assert per_scorer == {"Correctness": 0.8, "Safety": 0.6}  # Average of each scorer


def test_compute_per_scorer_scores_empty_results():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")
    assert optimizer._compute_per_scorer_scores([]) == {}


def test_compute_per_scorer_scores_no_individual_scores():
    optimizer = MetaPromptOptimizer(reflection_model="openai:/gpt-4o")

    eval_results = [
        EvaluationResultRecord(
            inputs={"q": "1"},
            outputs="a",
            expectations="a",
            score=0.8,
            trace=Mock(),
            rationales={},
            individual_scores={},
        ),
    ]

    assert optimizer._compute_per_scorer_scores(eval_results) == {}
