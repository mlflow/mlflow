from unittest import mock

import openai

import mlflow
from mlflow.utils.openai_utils import (
    TEST_CONTENT,
    _chat_completion_json_sample,
    _MockResponse,
)


def _mock_chat_completion_response(content=TEST_CONTENT):
    return _MockResponse(200, _chat_completion_json_sample(content))


def test_chat_completions_autolog():
    mlflow.openai.autolog(log_models=True)
    with mock.patch("mlflow.openai.log_model") as log_model_mock, mock.patch(
        "mlflow.tracking.MlflowClient.log_text"
    ) as log_text_mock:
        output = openai.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            temperature=0,
        )
        # ensure openai is mocked
        assert output.id == "chatcmpl-123"
        log_model_mock.assert_called_once()
        assert log_text_mock.call_count == 2
        assert log_text_mock.call_args_list[0].artifact_path.endswith("input.json")
        assert log_text_mock.call_args_list[1].artifact_path.endswith("output.json")
