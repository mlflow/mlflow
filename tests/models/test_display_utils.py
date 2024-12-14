from unittest import mock

import pytest

from mlflow.models import infer_signature
from mlflow.models.display_utils import should_render_agent_eval_template
from mlflow.types.llm import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from mlflow.models.rag_signatures import StringResponse


@pytest.fixture(scope="session", autouse=True)
def enable_databricks_env():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_runtime", return_value=True):
        with mock.patch("IPython.get_ipython", return_value=True):
            yield


def test_should_render_agent_eval_template_when_signature_is_chat_completion(enable_databricks_env):
    chat_request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="What is the primary function of control rods in a nuclear reactor?",
            ),
            ChatMessage(role="user", content="What is MLflow?"),
        ]
    )
    chat_response = ChatCompletionResponse(
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content="MLflow is an open source platform for the machine learning lifecycle.",
                ),
            )
        ]
    )
    signature = infer_signature(chat_request.to_dict(), chat_response.to_dict())
    assert should_render_agent_eval_template(signature)


def test_should_render_agent_eval_template_with_string_response(enable_databricks_env):
    chat_request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="What is the primary function of control rods in a nuclear reactor?",
            ),
            ChatMessage(role="user", content="What is MLflow?"),
        ]
    )
    string_response = StringResponse(
        content="MLflow is an open source platform for the machine learning lifecycle."
    )
    signature = infer_signature(chat_request.to_dict(), string_response)
    assert should_render_agent_eval_template(signature)
