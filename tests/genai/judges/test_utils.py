import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import _IS_LITELLM_INSTALLED, CategoricalRating, invoke_judge_model


def test_invoke_judge_model_successful_with_litellm():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    # Mock as if litellm is installed.
    mock_litellm = mock.MagicMock()
    with (
        mock.patch.dict("sys.modules", {"litellm": mock_litellm}),
        mock.patch.object(mlflow.genai.judges.utils, "_IS_LITELLM_INSTALLED", True),
    ):
        mock_litellm.completion.return_value.choices[0].message.content = mock_response

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    assert mock_litellm.completion.call_args.kwargs == {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "Evaluate this response"}],
    }

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


@pytest.mark.skipif(
    _IS_LITELLM_INSTALLED, reason="LiteLLM is installed, skipping native provider test"
)
def test_invoke_judge_model_successful_with_native_provider():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    with mock.patch(
        "mlflow.genai.judges.utils.score_model_on_payload", return_value=mock_response
    ) as mock_score_model_on_payload:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    assert mock_score_model_on_payload.call_args.kwargs == {
        "model_uri": "openai:/gpt-4",
        "payload": "Evaluate this response",
        "endpoint_type": "llm/v1/chat",
    }

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


@pytest.mark.skipif(
    _IS_LITELLM_INSTALLED, reason="LiteLLM is installed, skipping native provider test"
)
def test_invoke_judge_model_with_unsupported_provider():
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using 'unsupported' LLM"):
        invoke_judge_model(
            model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
        )


@pytest.mark.skipif(
    _IS_LITELLM_INSTALLED, reason="LiteLLM is installed, skipping native provider test"
)
def test_invoke_judge_model_unknown_rating_value():
    mock_response = json.dumps({"result": "maybe", "rationale": "Uncertain response"})

    with mock.patch("mlflow.genai.judges.utils.score_model_on_payload", return_value=mock_response):
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4", prompt="Evaluate", assessment_name="test"
        )

    assert feedback.value == CategoricalRating.UNKNOWN
    assert feedback.rationale == "Uncertain response"


@pytest.mark.skipif(
    _IS_LITELLM_INSTALLED, reason="LiteLLM is installed, skipping native provider test"
)
def test_invoke_judge_model_invalid_json_response():
    mock_response = "This is not valid JSON"

    with mock.patch("mlflow.genai.judges.utils.score_model_on_payload", return_value=mock_response):
        with pytest.raises(MlflowException, match=r"Failed to parse"):
            invoke_judge_model(
                model_uri="openai:/gpt-4", prompt="Test prompt", assessment_name="test"
            )
