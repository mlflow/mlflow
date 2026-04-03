import json
from unittest import mock

import pytest

from mlflow.entities.assessment import AssessmentError
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.custom_prompt_judge import _remove_choice_brackets, custom_prompt_judge

from tests.genai.conftest import databricks_only


def _mock_score(return_value):
    return mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        return_value=return_value,
    )


def test_custom_prompt_judge_basic():
    prompt_template = """Evaluate the response.

    <request>{{request}}</request>
    <response>{{response}}</response>

    Choose one:
    [[good]]: The response is good.
    [[bad]]: The response is bad.
    """

    mock_content = json.dumps({"result": "good", "rationale": "The response is well-written."})

    judge = custom_prompt_judge(
        name="quality", prompt_template=prompt_template, model="openai:/gpt-4"
    )

    with _mock_score(mock_content) as mock_score:
        feedback = judge(request="Test request", response="This is a great response!")

    assert feedback.name == "quality"
    assert feedback.value == "good"
    assert feedback.rationale == "The response is well-written."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "custom_prompt_judge_quality"

    mock_score.assert_called_once()
    kwargs = mock_score.call_args.kwargs
    assert kwargs["model_uri"] == "openai:/gpt-4"
    prompt = kwargs["payload"]
    assert prompt.startswith("Evaluate the response.")
    assert "<request>Test request</request>" in prompt
    assert "good: The response is good." in prompt
    assert "Answer ONLY in JSON and NOT in markdown," in prompt


@databricks_only
def test_custom_prompt_judge_databricks():
    prompt_template = """Evaluate the response.
    <request>{{request}}</request>
    Choose one:
    [[good]]: The response is good.
    """

    with mock.patch("databricks.agents.evals.judges.custom_prompt_judge") as mock_db_judge:
        custom_prompt_judge(name="quality", prompt_template=prompt_template, model="databricks")

    mock_db_judge.assert_called_once_with(
        name="quality", prompt_template=prompt_template, numeric_values=None
    )


def test_custom_prompt_judge_with_numeric_values():
    prompt_template = """
    Rate the response.

    <response>{{response}}</response>

    [[excellent]]: 5 stars
    [[great]]: 4 stars
    [[good]]: 3 stars
    [[not_good]]: 2 stars
    [[poor]]: 1 star
    """

    numeric_values = {"excellent": 5.0, "great": 4.0, "good": 3.0, "not_good": 2.0, "poor": 1.0}

    mock_content = json.dumps({"result": "good", "rationale": "Decent response."})

    judge = custom_prompt_judge(
        name="rating",
        prompt_template=prompt_template,
        numeric_values=numeric_values,
    )

    with _mock_score(mock_content) as mock_score:
        feedback = judge(response="This is okay.")

    assert feedback.name == "rating"
    assert feedback.value == 3.0
    assert feedback.metadata == {"string_value": "good"}
    assert feedback.rationale == "Decent response."

    mock_score.assert_called_once()
    kwargs = mock_score.call_args.kwargs
    assert kwargs["model_uri"] == "openai:/gpt-4.1-mini"
    prompt = kwargs["payload"]
    assert prompt.startswith("Rate the response.")
    assert '"rationale": "Reason for the decision.' in prompt


def test_custom_prompt_judge_no_choices_error():
    prompt_template = "Evaluate the response: {{response}}"

    with pytest.raises(ValueError, match="No choices found"):
        custom_prompt_judge(name="invalid", prompt_template=prompt_template)


def test_custom_prompt_judge_numeric_values_mismatch():
    prompt_template = """
    [[good]]: Good
    [[bad]]: Bad
    """

    numeric_values = {
        "good": 1.0,
        "bad": 0.0,
        "neutral": 0.5,  # Extra key not in choices
    }

    with pytest.raises(ValueError, match="numeric_values keys must match"):
        custom_prompt_judge(
            name="test", prompt_template=prompt_template, numeric_values=numeric_values
        )


def test_custom_prompt_judge_llm_error():
    prompt_template = """
    [[good]]: Good
    [[bad]]: Bad
    """

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        side_effect=Exception("API Error"),
    ):
        judge = custom_prompt_judge(name="test", prompt_template=prompt_template)

        feedback = judge(response="Test")

    assert feedback.name == "test"
    assert feedback.value is None
    assert isinstance(feedback.error, AssessmentError)
    assert "API Error" in feedback.error.error_message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Choose [[option1]] for the answer.", "Choose option1 for the answer."),
        (
            "Choose from [[formal]], [[informal]], or [[neutral]]",
            "Choose from formal, informal, or neutral",
        ),
        ("This text has no brackets.", "This text has no brackets."),
        # Single brackets are preserved
        ("Array[0] and [[choice1]] together.", "Array[0] and choice1 together."),
        # "-", "#" are not allowed in choice names
        (
            "Select [[option-1]], [[option_2]], or [[option#3]].",
            "Select [[option-1]], option_2, or [[option#3]].",
        ),
    ],
)
def test_remove_choice_brackets(text, expected):
    assert _remove_choice_brackets(text) == expected
