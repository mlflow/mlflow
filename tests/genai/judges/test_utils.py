import json
import pytest
from unittest import mock

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import invoke_judge_model, CategoricalRating, get_default_model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def test_invoke_judge_model_successful_with_rationale():
    mock_response = json.dumps({
        "result": "yes",
        "rationale": "The response meets all criteria."
    })

    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload",
        return_value=mock_response
    ) as mock_score_model:
        feedback = invoke_judge_model(
            model="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check"
        )

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_invoke_judge_model_unknown_rating_value():
    mock_response = json.dumps({
        "result": "maybe",
        "rationale": "Uncertain response"
    })

    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload",
        return_value=mock_response
    ):
        feedback = invoke_judge_model(
            model="custom:/model",
            prompt="Evaluate",
            assessment_name="test"
        )

    assert feedback.value == CategoricalRating.UNKNOWN
    assert feedback.rationale == "Uncertain response"


def test_invoke_judge_model_invocation_failure():
    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload",
        side_effect=Exception("API connection failed")
    ):
        with pytest.raises(MlflowException) as exc_info:
            invoke_judge_model(
                model="openai:/gpt-4",
                prompt="Test prompt",
                assessment_name="test"
            )

        assert "Failed to invoke the judge model" in str(exc_info.value)
        assert "Model: openai:/gpt-4" in str(exc_info.value)
        assert "Prompt: Test prompt" in str(exc_info.value)
        assert "API connection failed" in str(exc_info.value.__cause__)


def test_invoke_judge_model_invalid_json_response():
    mock_response = "This is not valid JSON"

    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload",
        return_value=mock_response
    ):
        with pytest.raises(MlflowException) as exc_info:
            invoke_judge_model(
                model="openai:/gpt-4",
                prompt="Test prompt",
                assessment_name="test"
            )

        assert "Failed to parse the response from the judge model" in str(exc_info.value)
        assert f"Response: {mock_response}" in str(exc_info.value)
        assert exc_info.value.error_code == INVALID_PARAMETER_VALUE
