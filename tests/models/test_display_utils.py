from pathlib import Path
from unittest import mock

import pytest

from mlflow.models import infer_signature
from mlflow.models.display_utils import (
    _generate_agent_eval_recipe,
    _should_render_agent_eval_template,
)
from mlflow.models.rag_signatures import StringResponse
from mlflow.types.llm import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

_CHAT_REQUEST = ChatCompletionRequest(
    messages=[
        ChatMessage(
            role="user",
            content="What is the primary function of control rods in a nuclear reactor?",
        ),
        ChatMessage(role="user", content="What is MLflow?"),
    ]
).to_dict()
_CHAT_RESPONSE = ChatCompletionResponse(
    choices=[
        ChatChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content="MLflow is an open source platform for the machine learning lifecycle.",
            ),
        )
    ]
).to_dict()

_STRING_RESPONSE = StringResponse(
    content="MLflow is an open source platform for the machine learning lifecycle."
)


@pytest.fixture
def enable_databricks_env():
    with (
        mock.patch("mlflow.utils.databricks_utils.is_in_databricks_runtime", return_value=True),
        mock.patch("IPython.get_ipython", return_value=True),
    ):
        yield


def test_should_render_eval_template_when_signature_is_chat_completion(enable_databricks_env):
    signature = infer_signature(_CHAT_REQUEST, _CHAT_RESPONSE)
    assert _should_render_agent_eval_template(signature)


def test_should_render_eval_template_with_string_response(enable_databricks_env):
    signature = infer_signature(_CHAT_REQUEST, _STRING_RESPONSE)
    assert _should_render_agent_eval_template(signature)


def test_should_render_eval_template_with_vanilla_string(enable_databricks_env):
    signature = infer_signature(_CHAT_REQUEST, "A vanilla string response")
    assert _should_render_agent_eval_template(signature)


def test_should_render_eval_template_with_string_input(enable_databricks_env):
    signature = infer_signature("A vanilla string input", _STRING_RESPONSE)
    assert _should_render_agent_eval_template(signature)


def test_should_not_render_eval_template_generic_signature(enable_databricks_env):
    signature = infer_signature({"input": "string"}, {"output": "string"})
    assert not _should_render_agent_eval_template(signature)


def test_should_not_render_eval_template_outside_databricks_env():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_runtime", return_value=False):
        with mock.patch("IPython.get_ipython", return_value=True):
            signature = infer_signature(_CHAT_REQUEST, _STRING_RESPONSE)
            assert not _should_render_agent_eval_template(signature)


def test_should_not_render_eval_template_outside_notebook_env():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_runtime", return_value=True):
        with mock.patch("IPython.get_ipython", return_value=None):
            signature = infer_signature(_CHAT_REQUEST, _STRING_RESPONSE)
            assert not _should_render_agent_eval_template(signature)


def test_generate_agent_eval_recipe():
    expected_html = (Path(__file__).parent / "resources" / "agent_eval_recipe.html").read_text()
    assert _generate_agent_eval_recipe("runs:/1/model") == expected_html
