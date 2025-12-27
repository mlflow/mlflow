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
from mlflow.exceptions import MlflowException
from mlflow.genai import judges
from mlflow.genai.evaluation.entities import EvalItem, EvalResult
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import RelevanceToQuery, Safety, Scorer, UserFrustration
from mlflow.genai.scorers.aggregation import compute_aggregated_metrics
from mlflow.genai.scorers.base import SerializedScorer
from mlflow.genai.scorers.builtin_scorers import _sanitize_scorer_feedback
from mlflow.genai.utils.type import FunctionCall
from mlflow.types.chat import ChatTool, FunctionToolDefinition

from tests.genai.conftest import databricks_only


def create_test_feedback(value: str, error: str | None = None) -> Feedback:
    return Feedback(
        name="test_feedback",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="databricks"),
        rationale="Test rationale",
        metadata={},
        value=value,
        error=error,
    )


def test_sanitize_scorer_feedback_happy_path():
    feedback = create_test_feedback("yes")
    result = _sanitize_scorer_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.YES


def test_sanitize_scorer_feedback_no():
    feedback = create_test_feedback("no")
    result = _sanitize_scorer_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.NO


def test_sanitize_scorer_feedback_unknown():
    feedback = create_test_feedback("unknown")
    result = _sanitize_scorer_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.UNKNOWN


def test_sanitize_scorer_feedback_error():
    feedback = create_test_feedback(None, error=AssessmentError(error_code="test_error"))
    result = _sanitize_scorer_feedback(feedback)
    assert result.value is None
    assert result.error == AssessmentError(error_code="test_error")


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("true", CategoricalRating.YES),
        ("True", CategoricalRating.YES),
        ("TRUE", CategoricalRating.YES),
        ("pass", CategoricalRating.YES),
        ("passed", CategoricalRating.YES),
        ("correct", CategoricalRating.YES),
        ("success", CategoricalRating.YES),
        ("1", CategoricalRating.YES),
        ("1.0", CategoricalRating.YES),
        ("false", CategoricalRating.NO),
        ("False", CategoricalRating.NO),
        ("FALSE", CategoricalRating.NO),
        ("fail", CategoricalRating.NO),
        ("failed", CategoricalRating.NO),
        ("incorrect", CategoricalRating.NO),
        ("failure", CategoricalRating.NO),
        ("0", CategoricalRating.NO),
        ("0.0", CategoricalRating.NO),
        ("maybe", CategoricalRating.UNKNOWN),
        ("partially", CategoricalRating.UNKNOWN),
        ("2", CategoricalRating.UNKNOWN),
        ("  yes  ", CategoricalRating.YES),
        ("  true  ", CategoricalRating.YES),
        ("  false  ", CategoricalRating.NO),
    ],
)
def test_sanitize_scorer_feedback_boolean_synonyms(input_value, expected):
    feedback = create_test_feedback(input_value)
    result = _sanitize_scorer_feedback(feedback)
    assert result.value == expected
    assert result.rationale == "Test rationale"


def test_sanitize_scorer_feedback_preserves_empty_string():
    feedback = Feedback(
        name="test_metric",
        value="",
        rationale="Test",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test-judge"),
        error="Empty value",
    )
    sanitized = _sanitize_scorer_feedback(feedback)
    assert sanitized.value == ""
    # String errors are converted to AssessmentError objects
    assert sanitized.error.error_message == "Empty value"
    assert sanitized.error.error_code == "ASSESSMENT_ERROR"


def test_sanitize_scorer_feedback_handles_categorical_rating_input():
    for rating in [CategoricalRating.YES, CategoricalRating.NO, CategoricalRating.UNKNOWN]:
        feedback = create_test_feedback(rating)
        sanitized = _sanitize_scorer_feedback(feedback)
        assert sanitized.value == rating


@pytest.mark.parametrize(
    ("values", "expected_mean"),
    [
        (["yes", "true", "pass", "correct", "1"], 1.0),
        (["no", "false", "fail", "incorrect", "0"], 0.0),
        (["yes", "no", "true", "false"], 0.5),
        (["pass", "fail", "1", "0"], 0.5),
        (["yes", "no", "maybe"], 0.5),
        (["true", "unknown", "false"], 0.5),
    ],
)
def test_aggregation_with_sanitized_boolean_values(values, expected_mean):
    eval_results = []
    for i, value in enumerate(values):
        feedback = create_test_feedback(value)
        sanitized = _sanitize_scorer_feedback(feedback)
        eval_item = EvalItem(
            inputs={"question": f"Q{i}"},
            outputs={"answer": f"A{i}"},
            request_id=f"req_{i}",
            expectations={},
        )
        eval_result = EvalResult(assessments=[sanitized], eval_item=eval_item)
        eval_results.append(eval_result)

    scorer = mock.Mock(spec=Scorer)
    scorer.name = "test_feedback"
    scorer.aggregations = ["mean"]

    metrics = compute_aggregated_metrics(eval_results, [scorer])
    assert "test_feedback/mean" in metrics
    assert metrics["test_feedback/mean"] == pytest.approx(expected_mean)


def test_aggregation_excludes_unknown_boolean_values():
    values = ["yes", "no", "maybe", "partially", "true", "false", "unknown"]
    eval_results = []
    for i, value in enumerate(values):
        feedback = create_test_feedback(value)
        sanitized = _sanitize_scorer_feedback(feedback)
        eval_item = EvalItem(
            inputs={"question": f"Q{i}"},
            outputs={"answer": f"A{i}"},
            request_id=f"req_{i}",
            expectations={},
        )
        eval_result = EvalResult(assessments=[sanitized], eval_item=eval_item)
        eval_results.append(eval_result)

    scorer = mock.Mock(spec=Scorer)
    scorer.name = "test_feedback"
    scorer.aggregations = ["mean", "min", "max"]

    metrics = compute_aggregated_metrics(eval_results, [scorer])
    assert metrics["test_feedback/mean"] == 0.5
    assert metrics["test_feedback/min"] == 0.0
    assert metrics["test_feedback/max"] == 1.0


def test_builtin_scorer_handles_boolean_synonyms():
    with mock.patch("mlflow.genai.judges.is_context_relevant") as mock_judge:
        mock_feedback = Feedback(
            name="relevance_to_query",
            value="true",
            rationale="The context is relevant",
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
        )
        mock_judge.return_value = mock_feedback

        scorer = RelevanceToQuery()
        result = scorer(
            inputs={"question": "What is the capital of France?"},
            outputs="Paris is the capital of France.",
        )
        assert result.value == CategoricalRating.YES
        assert result.rationale == "The context is relevant"


def test_builtin_scorer_handles_numeric_boolean_values():
    with mock.patch("mlflow.genai.judges.is_context_relevant") as mock_judge:
        for input_val, expected in [("1", CategoricalRating.YES), ("0", CategoricalRating.NO)]:
            mock_feedback = Feedback(
                name="relevance_to_query",
                value=input_val,
                rationale="Test rationale",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"
                ),
            )
            mock_judge.return_value = mock_feedback

            scorer = RelevanceToQuery()
            result = scorer(inputs={"question": "Test question"}, outputs="Test context")
            assert result.value == expected


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


def test_is_correct_rejects_both_expected_response_and_expected_facts():
    with pytest.raises(
        MlflowException,
        match="Only one of expected_response or expected_facts should be provided, not both",
    ):
        judges.is_correct(
            request="What is the capital of France?",
            response="Paris is the capital of France.",
            expected_response="Paris",
            expected_facts=["Paris is the capital of France"],
        )


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
@databricks_only
def test_judge_functions_databricks(judge_func, agents_judge_name, args):
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
@databricks_only
def test_judge_functions_called_with_correct_name(name, expected_name):
    with mock.patch("databricks.agents.evals.judges.relevance_to_query") as mock_judge:
        judges.is_context_relevant(request="test", context="test", name=name)
        mock_judge.assert_called_once_with(
            request="test",
            response="test",
            assessment_name=expected_name,
        )


def test_is_safe_oss_with_custom_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(
            name="safety",
            value=CategoricalRating.YES,
            rationale="The content is safe and appropriate.",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="anthropic:/claude-3-sonnet"
            ),
        ),
    ) as mock_invoke:
        feedback = judges.is_safe(
            content="This is a safe message",
            model="anthropic:/claude-3-sonnet",
        )

    assert feedback.name == "safety"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The content is safe and appropriate."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "anthropic:/claude-3-sonnet"

    mock_invoke.assert_called_once()
    args, kwargs = mock_invoke.call_args
    assert args[0] == "anthropic:/claude-3-sonnet"  # model
    assert kwargs["assessment_name"] == "safety"


def test_is_safe_with_custom_name_and_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(
            name="custom_safety_check",
            value=CategoricalRating.NO,
            rationale="The content may be inappropriate.",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="openai:/gpt-4-turbo"
            ),
        ),
    ) as mock_invoke:
        feedback = judges.is_safe(
            content="Some potentially unsafe content",
            name="custom_safety_check",
            model="openai:/gpt-4-turbo",
        )

    assert feedback.name == "custom_safety_check"
    assert feedback.value == CategoricalRating.NO
    assert feedback.rationale == "The content may be inappropriate."
    assert feedback.source.source_id == "openai:/gpt-4-turbo"

    mock_invoke.assert_called_once()
    args, kwargs = mock_invoke.call_args
    assert args[0] == "openai:/gpt-4-turbo"  # model
    assert kwargs["assessment_name"] == "custom_safety_check"


@databricks_only
def test_is_safe_databricks_with_custom_model():
    # When model is "databricks", should still use databricks judge
    with mock.patch(
        "databricks.agents.evals.judges.safety",
        return_value=Feedback(
            name="safety",
            value=judges.CategoricalRating.YES,
            rationale="Safe content.",
        ),
    ) as mock_safety:
        result = judges.is_safe(
            content="Test content",
            model="databricks",  # Explicitly use databricks
        )

        assert isinstance(result.value, judges.CategoricalRating)
        assert result.value == judges.CategoricalRating.YES
        mock_safety.assert_called_once_with(
            response="Test content",
            assessment_name="safety",
        )


def test_ser_deser():
    judge = Safety()
    serialized1 = judge.model_dump()
    serialized2 = SerializedScorer(**serialized1)
    for serialized in [serialized1, serialized2]:
        deserialized = Scorer.model_validate(serialized)
        assert isinstance(deserialized, Safety)
        assert deserialized.name == "safety"
        assert deserialized.required_columns == {"inputs", "outputs"}


def test_ser_deser_session_level_scorer():
    scorer = UserFrustration()

    # Verify the scorer is session-level
    assert scorer.is_session_level_scorer is True

    # Test serialization
    serialized_dict = scorer.model_dump()
    assert serialized_dict["is_session_level_scorer"] is True
    assert serialized_dict["name"] == "user_frustration"
    assert serialized_dict["builtin_scorer_class"] == "UserFrustration"

    # Test deserialization from dict
    deserialized = Scorer.model_validate(serialized_dict)
    assert isinstance(deserialized, UserFrustration)
    assert deserialized.name == "user_frustration"
    assert deserialized.is_session_level_scorer is True

    # Test deserialization from SerializedScorer object
    serialized_obj = SerializedScorer(**serialized_dict)
    deserialized2 = Scorer.model_validate(serialized_obj)
    assert isinstance(deserialized2, UserFrustration)
    assert deserialized2.is_session_level_scorer is True


def test_is_tool_call_efficient_with_custom_name_and_model():
    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(
            name="custom_efficiency_check",
            value=CategoricalRating.YES,
            rationale="Let's think step by step. Tool usage is optimal.",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="anthropic:/claude-3-sonnet",
            ),
        ),
    ) as mock_invoke:
        feedback = judges.is_tool_call_efficient(
            request="Get weather for Paris",
            tools_called=[
                FunctionCall(
                    name="get_weather",
                    arguments={"city": "Paris"},
                    outputs="Sunny, 22°C",
                    exception=None,
                )
            ],
            available_tools=[
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(name="get_weather", description="Get weather"),
                )
            ],
            name="custom_efficiency_check",
            model="anthropic:/claude-3-sonnet",
        )

    assert feedback.name == "custom_efficiency_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.source.source_id == "anthropic:/claude-3-sonnet"

    mock_invoke.assert_called_once()
    args, kwargs = mock_invoke.call_args
    assert args[0] == "anthropic:/claude-3-sonnet"
    assert kwargs["assessment_name"] == "custom_efficiency_check"


def test_is_tool_call_correct_with_custom_name_and_model():
    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(
            name="custom_correctness_check",
            value=CategoricalRating.YES,
            rationale="Let's think step by step. Tool calls and arguments are appropriate.",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="anthropic:/claude-3-sonnet",
            ),
        ),
    ) as mock_invoke:
        feedback = judges.is_tool_call_correct(
            request="Get weather for Paris",
            tools_called=[
                FunctionCall(
                    name="get_weather",
                    arguments={"city": "Paris"},
                    outputs="Sunny, 22°C",
                )
            ],
            available_tools=[
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(name="get_weather", description="Get weather"),
                )
            ],
            name="custom_correctness_check",
            model="anthropic:/claude-3-sonnet",
        )

    assert feedback.name == "custom_correctness_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.source.source_id == "anthropic:/claude-3-sonnet"

    mock_invoke.assert_called_once()
    args, kwargs = mock_invoke.call_args
    assert args[0] == "anthropic:/claude-3-sonnet"
    assert kwargs["assessment_name"] == "custom_correctness_check"
