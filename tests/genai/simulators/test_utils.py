from unittest import mock

import pytest

from mlflow.genai.simulators.utils import format_history, invoke_model_without_tracing


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], None),
        ([{"role": "user", "content": "Hello"}], "user: Hello"),
        (
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "user: Hello\nassistant: Hi there!\nuser: How are you?",
        ),
        ([{"content": "Hello"}], "unknown: Hello"),
        ([{"role": "user"}], "user: "),
        ([{"role": None, "content": None}], "unknown: "),
    ],
)
def test_format_history(history, expected):
    assert format_history(history) == expected


@pytest.mark.parametrize(
    "model_uri",
    [
        "openai:/gpt-4o-mini",
        "anthropic:/claude-3-haiku",
    ],
)
def test_invoke_model_without_tracing_with_litellm(model_uri):
    from mlflow.types.llm import ChatMessage

    messages = [ChatMessage(role="user", content="Hello")]

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="Hi there!"))]
        )

        result = invoke_model_without_tracing(model_uri=model_uri, messages=messages)

        assert result == "Hi there!"
        mock_completion.assert_called_once()


def test_invoke_model_without_tracing_with_inference_params():
    from mlflow.types.llm import ChatMessage

    messages = [ChatMessage(role="user", content="Hello")]

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="Response"))]
        )

        invoke_model_without_tracing(
            model_uri="openai:/gpt-4o-mini",
            messages=messages,
            inference_params={"temperature": 0.5},
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5


def test_invoke_model_without_tracing_with_databricks():
    from mlflow.types.llm import ChatMessage

    messages = [ChatMessage(role="user", content="Hello")]

    with (
        mock.patch("mlflow.genai.simulators.utils.call_chat_completions") as mock_call,
        mock.patch(
            "mlflow.genai.simulators.utils.create_litellm_message_from_databricks_response"
        ) as mock_create,
    ):
        mock_call.return_value = mock.MagicMock(error_code=None, output_json='{"content": "Hi"}')
        mock_create.return_value = mock.MagicMock(content="Hi from Databricks")

        result = invoke_model_without_tracing(model_uri="databricks", messages=messages)

        assert result == "Hi from Databricks"
        mock_call.assert_called_once()
