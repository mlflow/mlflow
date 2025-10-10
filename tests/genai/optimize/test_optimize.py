from unittest.mock import Mock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.genai.optimize.optimize import _make_output_equivalence_scorer, optimize_prompts
from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput
from mlflow.genai.prompts import register_prompt
from mlflow.genai.scorers import scorer


class MockPromptAdapter(BasePromptOptimizer):
    def __init__(self, reflection_model="openai:/gpt-4o-mini"):
        self.model_name = reflection_model

    def optimize(self, eval_fn, train_data, target_prompts):
        optimized_prompts = {}
        for prompt_name, template in target_prompts.items():
            # Simple optimization: add "Be precise and accurate. " prefix
            optimized_prompts[prompt_name] = f"Be precise and accurate. {template}"

        # Verify the optimization by calling eval_fn
        eval_fn(optimized_prompts, train_data)

        return PromptOptimizerOutput(
            optimized_prompts=optimized_prompts,
            initial_eval_score=0.5,
            final_eval_score=0.9,
        )


@pytest.fixture
def sample_translation_prompt():
    return register_prompt(
        name="test_translation_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
    )


@pytest.fixture
def sample_summarization_prompt():
    return register_prompt(
        name="test_summarization_prompt",
        template="Summarize this text: {{text}}",
    )


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "inputs": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
                {"input_text": "Goodbye", "language": "Spanish"},
            ],
            "outputs": [
                "Hola",
                "Monde",
                "Adiós",
            ],
        }
    )


@pytest.fixture
def sample_summarization_dataset():
    return [
        {
            "inputs": {
                "text": "This is a long document that needs to be summarized into key points."
            },
            "outputs": "Key points summary",
        },
        {
            "inputs": {"text": "Another document with important information for summarization."},
            "outputs": "Important info summary",
        },
    ]


def sample_predict_fn(input_text: str, language: str) -> str:
    mlflow.genai.load_prompt("prompts:/test_translation_prompt/1")
    translations = {
        ("Hello", "Spanish"): "Hola",
        ("World", "French"): "Monde",
        ("Goodbye", "Spanish"): "Adiós",
    }
    return translations.get((input_text, language), f"translated_{input_text}")


def sample_summarization_fn(text):
    return f"Summary of: {text[:20]}..."


def test_adapt_prompts_single_prompt(sample_translation_prompt, sample_dataset):
    mock_adapter = MockPromptAdapter()

    result = optimize_prompts(
        predict_fn=sample_predict_fn,
        train_data=sample_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}"
        ],
        optimizer=mock_adapter,
        scorers=[_make_output_equivalence_scorer(mock_adapter.model_name)],
    )

    assert len(result.optimized_prompts) == 1
    optimized_prompt = result.optimized_prompts[0]
    assert optimized_prompt.name == sample_translation_prompt.name
    assert optimized_prompt.version == sample_translation_prompt.version + 1
    assert "Be precise and accurate." in optimized_prompt.template
    expected_template = "Translate the following text to {{language}}: {{input_text}}"
    assert expected_template in optimized_prompt.template
    assert result.initial_eval_score == 0.5
    assert result.final_eval_score == 0.9


def test_adapt_prompts_multiple_prompts(
    sample_translation_prompt, sample_summarization_prompt, sample_dataset
):
    mock_adapter = MockPromptAdapter()

    result = optimize_prompts(
        predict_fn=sample_predict_fn,
        train_data=sample_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}",
            f"prompts:/{sample_summarization_prompt.name}/{sample_summarization_prompt.version}",
        ],
        optimizer=mock_adapter,
        scorers=[_make_output_equivalence_scorer(mock_adapter.model_name)],
    )

    assert len(result.optimized_prompts) == 2
    prompt_names = {prompt.name for prompt in result.optimized_prompts}
    assert sample_translation_prompt.name in prompt_names
    assert sample_summarization_prompt.name in prompt_names
    assert result.initial_eval_score == 0.5
    assert result.final_eval_score == 0.9

    for prompt in result.optimized_prompts:
        assert "Be precise and accurate." in prompt.template


def test_adapt_prompts_eval_function_behavior(sample_translation_prompt, sample_dataset):
    class TestingAdapter(BasePromptOptimizer):
        def __init__(self):
            self.model_name = "openai:/gpt-4o-mini"
            self.eval_fn_calls = []

        def optimize(self, eval_fn, dataset, target_prompts):
            # Test that eval_fn works correctly
            test_prompts = {
                "test_translation_prompt": "Prompt Candidate: "
                "Translate {{input_text}} to {{language}}"
            }
            results = eval_fn(test_prompts, dataset)
            self.eval_fn_calls.append((test_prompts, results))

            # Verify results structure
            assert isinstance(results, list)
            assert len(results) == len(dataset)
            for i, result in enumerate(results):
                assert isinstance(result, EvaluationResultRecord)
                assert result.inputs == dataset[i]["inputs"]
                assert result.outputs == dataset[i]["outputs"]
                assert result.score == 1
                assert result.trace is not None

            return PromptOptimizerOutput(optimized_prompts=target_prompts)

    predict_called_count = 0

    def predict_fn(input_text, language):
        prompt = mlflow.genai.load_prompt("prompts:/test_translation_prompt/1").format(
            input_text=input_text, language=language
        )
        nonlocal predict_called_count
        # the first call to the predict_fn is the model check
        if predict_called_count > 0:
            # validate the prompt is replaced with the candidate prompt
            assert "Prompt Candidate" in prompt
        predict_called_count += 1

        return sample_predict_fn(input_text=input_text, language=language)

    testing_adapter = TestingAdapter()

    optimize_prompts(
        predict_fn=predict_fn,
        train_data=sample_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}"
        ],
        optimizer=testing_adapter,
        scorers=[_make_output_equivalence_scorer(testing_adapter.model_name)],
    )

    assert len(testing_adapter.eval_fn_calls) == 1
    _, eval_results = testing_adapter.eval_fn_calls[0]
    assert len(eval_results) == 3  # Number of records in sample_dataset
    assert predict_called_count == 4  # 3 records in sample_dataset + 1 for the prediction check


def test_adapt_prompts_with_list_dataset(sample_translation_prompt, sample_summarization_dataset):
    mock_adapter = MockPromptAdapter()

    def summarization_predict_fn(text):
        return f"Summary: {text[:10]}..."

    result = optimize_prompts(
        predict_fn=summarization_predict_fn,
        train_data=sample_summarization_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}"
        ],
        optimizer=mock_adapter,
        scorers=[_make_output_equivalence_scorer(mock_adapter.model_name)],
    )

    assert len(result.optimized_prompts) == 1
    assert result.initial_eval_score == 0.5
    assert result.final_eval_score == 0.9


def test_adapt_prompts_with_model_name(sample_translation_prompt, sample_dataset):
    class TestAdapter(BasePromptOptimizer):
        def __init__(self):
            self.model_name = "test/custom-model"

        def optimize(self, eval_fn, dataset, target_prompts):
            return PromptOptimizerOutput(optimized_prompts=target_prompts)

    testing_adapter = TestAdapter()

    result = optimize_prompts(
        predict_fn=sample_predict_fn,
        train_data=sample_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}"
        ],
        optimizer=testing_adapter,
        scorers=[_make_output_equivalence_scorer(testing_adapter.model_name)],
    )

    assert len(result.optimized_prompts) == 1


@pytest.mark.parametrize(
    ("program_outputs", "expected_outputs", "expected_score"),
    [
        # Numeric exact matches
        (42, 42, 1.0),
        (42, 43, 0.0),
        (3.14, 3.14, 1.0),
        (3.14, 3.15, 0.0),
        (True, True, 1.0),
        (True, False, 0.0),
        # Mixed numeric types
        (1, 1.0, 1.0),
        (0, False, 1.0),
        (1, True, 1.0),
        # String exact matches
        ("hello", "hello", 1.0),
        ("Paris", "Paris", 1.0),
        # Non-string types converted to strings
        ([1, 2, 3], [1, 2, 3], 1.0),
    ],
)
def test_output_equivalence_scorer_exact_match(program_outputs, expected_outputs, expected_score):
    test_scorer = _make_output_equivalence_scorer("openai:/gpt-4o-mini")
    assert (
        test_scorer.run(inputs={}, outputs=program_outputs, expectations=expected_outputs)
        == expected_score
    )


def test_output_equivalence_scorer_llm_judge():
    # Test pass case
    mock_pass = Mock(value="pass")
    with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
        mock_judge = Mock(return_value=mock_pass)
        mock_make_judge.return_value = mock_judge

        test_scorer = _make_output_equivalence_scorer("openai:/gpt-4o-mini")
        score = test_scorer.run(
            inputs={}, outputs="The capital of France is Paris", expectations="Paris"
        )

        # Verify correct parameters passed to make_judge
        assert mock_make_judge.call_args.kwargs["name"] == "equivalence_judge"
        assert mock_make_judge.call_args.kwargs["model"] == "openai:/gpt-4o-mini"
        assert "{{outputs}}" in mock_make_judge.call_args.kwargs["instructions"]
        assert "{{expectations}}" in mock_make_judge.call_args.kwargs["instructions"]

        # Verify judge called with string representations
        mock_judge.assert_called_once_with(
            outputs={"outputs": "The capital of France is Paris"},
            expectations={"outputs": "Paris"},
        )
        assert score == 1.0

    # Test fail case
    mock_result = Mock(value="fail")
    with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
        mock_judge = Mock(return_value=mock_result)
        mock_make_judge.return_value = mock_judge
        test_scorer = _make_output_equivalence_scorer("openai:/gpt-4o-mini")
        assert test_scorer.run(inputs={}, outputs="output", expectations="different") == 0.0


def test_output_equivalence_scorer_error_handling():
    with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
        mock_judge = Mock(side_effect=Exception("API Error"))
        mock_make_judge.return_value = mock_judge
        test_scorer = _make_output_equivalence_scorer("openai:/gpt-4o-mini")
        assert test_scorer.run(inputs={}, outputs="output", expectations="expected") == 0.0


def test_adapt_prompts_with_custom_scorers(sample_translation_prompt, sample_dataset):
    # Create a custom scorer for case-insensitive matching
    @scorer(name="case_insensitive_match")
    def case_insensitive_match(outputs, expectations):
        # Extract expected_response if expectations is a dict
        if isinstance(expectations, dict) and "expected_response" in expectations:
            expected_value = expectations["expected_response"]
        else:
            expected_value = expectations
        return 1.0 if str(outputs).lower() == str(expected_value).lower() else 0.5

    class MetricTestAdapter(BasePromptOptimizer):
        def __init__(self):
            self.model_name = "openai:/gpt-4o-mini"
            self.captured_scores = []

        def optimize(self, eval_fn, dataset, target_prompts):
            # Run eval_fn and capture the scores
            results = eval_fn(target_prompts, dataset)
            self.captured_scores = [r.score for r in results]
            return PromptOptimizerOutput(optimized_prompts=target_prompts)

    testing_adapter = MetricTestAdapter()

    # Create dataset with outputs that will test custom scorer
    test_dataset = pd.DataFrame(
        {
            "inputs": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
            ],
            "outputs": ["HOLA", "monde"],  # Different cases to test custom scorer
        }
    )

    def predict_fn(input_text, language):
        mlflow.genai.load_prompt("prompts:/test_translation_prompt/1")
        # Return lowercase outputs
        return {"Hello": "hola", "World": "monde"}.get(input_text, "unknown")

    result = optimize_prompts(
        predict_fn=predict_fn,
        train_data=test_dataset,
        prompt_uris=[
            f"prompts:/{sample_translation_prompt.name}/{sample_translation_prompt.version}"
        ],
        scorers=[case_insensitive_match],
        optimizer=testing_adapter,
    )

    # Verify custom scorer was used
    # "hola" vs "HOLA" (case insensitive match) -> 1.0
    # "monde" vs "monde" (exact match) -> 1.0
    assert testing_adapter.captured_scores == [1.0, 1.0]
    assert len(result.optimized_prompts) == 1
