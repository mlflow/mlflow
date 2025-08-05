from unittest import mock

import pytest

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai import judges
from mlflow.genai.judges.utils import CategoricalRating


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def _get_mock_response(text: str):
    return {
        "id": "3cdb958c-e4cc-4834-b52b-1d1a7f324714",
        "object": "chat.completion",
        "created": 1700173217,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {"content": text, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def test_guidelines():
    # mock openai
    mock_response = """\
{
    "result": "yes",
    "rationale": "The response is correct."
}"""
    with mock.patch(
        "mlflow.metrics.genai.model_utils._send_request",
        return_value=_get_mock_response(mock_response),
    ) as mock_request:
        feedback = judges.meets_guidelines(
            guidelines="The response must be in English.",
            context={"request": "What is the capital of France?", "response": "Paris"},
        )

    assert feedback.name == "guidelines"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response is correct."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4.1-mini"

    assert mock_request.call_count == 1
