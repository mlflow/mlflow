import json
from unittest import mock

import pytest
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import (
    AssessmentError,
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
)
from mlflow.genai import judges
from mlflow.genai.judges.builtin import _sanitize_feedback
from mlflow.genai.judges.utils import CategoricalRating


def create_test_feedback(value: str, error: str | None = None) -> Feedback:
    return Feedback(
        name="test_feedback",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="databricks"),
        rationale="Test rationale",
        metadata={},
        value=value,
        error=error,
    )


def test_sanitize_feedback_happy_path():
    feedback = create_test_feedback("yes")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.YES


def test_sanitize_feedback_no():
    feedback = create_test_feedback("no")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.NO


def test_sanitize_feedback_unknown():
    feedback = create_test_feedback("unknown")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.UNKNOWN


def test_sanitize_feedback_error():
    feedback = create_test_feedback(None, error=AssessmentError(error_code="test_error"))
    result = _sanitize_feedback(feedback)
    assert result.value is None
    assert result.error == AssessmentError(error_code="test_error")


def test_meets_guidelines_oss():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "Let's think step by step. The response is correct.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = judges.meets_guidelines(
            guidelines="The response must be in English.",
            context={"request": "What is the capital of France?", "response": "Paris"},
        )

    assert feedback.name == "guidelines"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response is correct."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_litellm.call_count == 1
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert kwargs["messages"][0]["role"] == "user"
    prompt = kwargs["messages"][0]["content"]
    assert prompt.startswith("Given the following set of guidelines and some inputs")
    assert "What is the capital of France?" in prompt


def test_is_context_relevant_oss():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "Let's think step by step. The answer is relevant to the question.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = judges.is_context_relevant(
            request="What is the capital of France?",
            context="Paris is the capital of France.",
        )

    assert feedback.name == "relevance_to_context"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The answer is relevant to the question."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_litellm.call_count == 1
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert kwargs["messages"][0]["role"] == "user"
    prompt = kwargs["messages"][0]["content"]
    assert "Consider the following question and answer" in prompt
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt


def test_is_correct_oss():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "Let's think step by step. The response is correct.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = judges.is_correct(
            request="What is the capital of France?",
            response="Paris is the capital of France.",
            expected_response="Paris",
        )

    assert feedback.name == "correctness"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response is correct."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_litellm.call_count == 1
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert kwargs["messages"][0]["role"] == "user"
    prompt = kwargs["messages"][0]["content"]
    assert "Consider the following question, claim and document" in prompt
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
    assert "Paris" in prompt


def test_is_context_sufficient_oss():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "Let's think step by step. The context is sufficient.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = judges.is_context_sufficient(
            request="What is the capital of France?",
            context=[
                {"content": "Paris is the capital of France."},
                {"content": "Paris is known for its Eiffel Tower."},
            ],
            expected_facts=["Paris is the capital of France."],
        )

    assert feedback.name == "context_sufficiency"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The context is sufficient."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_litellm.call_count == 1
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert kwargs["messages"][0]["role"] == "user"
    prompt = kwargs["messages"][0]["content"]
    assert "Consider the following claim and document" in prompt
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt


def test_is_grounded_oss():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "Let's think step by step. The response is grounded.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = judges.is_grounded(
            request="What is the capital of France?",
            response="Paris",
            context=[
                {"content": "Paris is the capital of France."},
                {"content": "Paris is known for its Eiffel Tower."},
            ],
        )

    assert feedback.name == "groundedness"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response is grounded."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_litellm.call_count == 1
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert kwargs["messages"][0]["role"] == "user"
    prompt = kwargs["messages"][0]["content"]
    assert "Consider the following claim and document" in prompt
    assert "What is the capital of France?" in prompt
    assert "Paris" in prompt
    assert "Paris is the capital of France." in prompt


@pytest.mark.parametrize(
    ("judge_func", "agents_judge_name", "args"),
    [
        (
            judges.is_context_relevant,
            "relevance_to_query",
            {"request": "test", "context": "test"},
        ),
        (
            judges.is_context_sufficient,
            "context_sufficiency",
            {"request": "test", "context": "test", "expected_facts": ["test"]},
        ),
        (
            judges.is_correct,
            "correctness",
            {"request": "test", "response": "test", "expected_facts": ["test"]},
        ),
        (
            judges.is_grounded,
            "groundedness",
            {"request": "test", "response": "test", "context": "test"},
        ),
        (
            judges.is_safe,
            "safety",
            {"content": "test"},
        ),
        (
            judges.meets_guidelines,
            "guidelines",
            {"guidelines": "test", "context": {"response": "test"}},
        ),
    ],
)
def test_judge_functions_databricks(judge_func, agents_judge_name, args, databricks_tracking_uri):
    with mock.patch(f"databricks.agents.evals.judges.{agents_judge_name}") as mock_judge:
        mock_judge.return_value = Feedback(
            name=agents_judge_name,
            value=judges.CategoricalRating.YES,
            rationale="The response is correct.",
        )
        result = judge_func(**args)
        assert isinstance(result.value, judges.CategoricalRating)
        assert result.value == judges.CategoricalRating.YES
        mock_judge.assert_called_once()


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [
        (None, "relevance_to_context"),
        ("test", "test"),
    ],
)
def test_judge_functions_called_with_correct_name(name, expected_name, databricks_tracking_uri):
    with mock.patch("databricks.agents.evals.judges.relevance_to_query") as mock_judge:
        judges.is_context_relevant(request="test", context="test", name=name)
        mock_judge.assert_called_once_with(
            request="test",
            response="test",
            assessment_name=expected_name,
        )
