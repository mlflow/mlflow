import json
from typing import Any
from uuid import uuid4

import pydantic
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.model import Model
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import load_serving_example
from mlflow.pyfunc.loaders.chat_agent import _ChatAgentPyfuncWrapper
from mlflow.pyfunc.model import ChatAgent
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.agent import (
    CHAT_AGENT_INPUT_EXAMPLE,
    CHAT_AGENT_INPUT_SCHEMA,
    CHAT_AGENT_OUTPUT_SCHEMA,
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.types.schema import ColSpec, DataType, Schema

from tests.helper_functions import (
    expect_status_code,
    pyfunc_serve_and_score_model,
)
from tests.tracing.helper import get_traces


def get_mock_response(messages: list[ChatAgentMessage], message=None):
    return {
        "messages": [
            {
                "role": "assistant",
                "content": message or msg.content,
                "name": "llm",
                "id": str(uuid4()),
            }
            for msg in messages
        ],
    }


class SimpleChatAgent(ChatAgent):
    @mlflow.trace
    def predict(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ) -> ChatAgentResponse:
        mock_response = get_mock_response(messages)
        return ChatAgentResponse(**mock_response)

    def predict_stream(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ):
        for i in range(5):
            mock_response = get_mock_response(messages, f"message {i}")
            mock_response["delta"] = mock_response["messages"][0]
            mock_response["delta"]["id"] = str(i)
            yield ChatAgentChunk(**mock_response)


class SimpleBadChatAgent(ChatAgent):
    @mlflow.trace
    def predict(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ) -> ChatAgentResponse:
        mock_response = get_mock_response(messages)
        return ChatAgentResponse(messages=mock_response)

    def predict_stream(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ):
        for i in range(5):
            mock_response = get_mock_response(messages, f"message {i}")
            mock_response["delta"] = mock_response["messages"][0]
            yield ChatAgentChunk(delta=mock_response)


class SimpleDictChatAgent(ChatAgent):
    @mlflow.trace
    def predict(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ) -> ChatAgentResponse:
        mock_response = get_mock_response(messages)
        return ChatAgentResponse(**mock_response).model_dump_compat()


class ChatAgentWithCustomInputs(ChatAgent):
    def predict(
        self, messages: list[ChatAgentMessage], context: ChatContext, custom_inputs: dict[str, Any]
    ) -> ChatAgentResponse:
        mock_response = get_mock_response(messages)
        return ChatAgentResponse(
            **mock_response,
            custom_outputs=custom_inputs,
        )


def test_chat_agent_save_load(tmp_path):
    model = SimpleChatAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatAgentPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == CHAT_AGENT_INPUT_SCHEMA
    assert output_schema == CHAT_AGENT_OUTPUT_SCHEMA


def test_chat_agent_save_load_dict_output(tmp_path):
    model = SimpleDictChatAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatAgentPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == CHAT_AGENT_INPUT_SCHEMA
    assert output_schema == CHAT_AGENT_OUTPUT_SCHEMA


def test_chat_agent_trace(tmp_path):
    model = SimpleChatAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    # predict() call during saving chat model should not generate a trace
    assert len(get_traces()) == 0

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    messages = [{"role": "user", "content": "Hello!"}]
    loaded_model.predict({"messages": messages})

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.tags[TraceTagKey.TRACE_NAME] == "predict"
    request = json.loads(traces[0].data.request)
    assert [{k: v for k, v in msg.items() if k != "id"} for msg in request["messages"]] == [
        {k: v for k, v in ChatAgentMessage(**msg).model_dump_compat().items() if k != "id"}
        for msg in messages
    ]


def test_chat_agent_save_throws_with_signature(tmp_path):
    model = SimpleChatAgent()

    with pytest.raises(MlflowException, match="Please remove the `signature` parameter"):
        mlflow.pyfunc.save_model(
            python_model=model,
            path=tmp_path,
            signature=ModelSignature(
                inputs=Schema([ColSpec(name="test", type=DataType.string)]),
            ),
        )


@pytest.mark.parametrize(
    "ret",
    [
        "not a ChatAgentResponse",
        {"dict": "with", "bad": "keys"},
        {
            "id": "1",
            "created": 1,
            "model": "m",
            "choices": [{"bad": "choice"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        },
    ],
)
def test_save_throws_on_invalid_output(tmp_path, ret):
    class BadChatAgent(ChatAgent):
        def predict(
            self,
            messages: list[ChatAgentMessage],
            context: ChatContext,
            custom_inputs: dict[str, Any],
        ) -> ChatAgentResponse:
            return ret

    model = BadChatAgent()
    with pytest.raises(
        MlflowException,
        match=("Failed to save ChatAgent. Ensure your model's predict"),
    ):
        mlflow.pyfunc.save_model(python_model=model, path=tmp_path)


def test_chat_agent_predict(tmp_path):
    model = ChatAgentWithCustomInputs()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)

    # test that a single dictionary will work
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]

    response = loaded_model.predict({"messages": messages})
    assert response["messages"][0]["content"] == "You are a helpful assistant"


def test_chat_agent_works_with_infer_signature_input_example():
    model = SimpleChatAgent()
    input_example = {
        "messages": [
            {
                "role": "system",
                "content": "You are in helpful assistant!",
            },
            {
                "role": "user",
                "content": "What is Retrieval-augmented Generation?",
            },
        ],
        "context": {
            "conversation_id": "123",
            "user_id": "456",
        },
        "stream": False,  # this is set by default
    }
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=model, input_example=input_example
        )
    assert model_info.signature.inputs == CHAT_AGENT_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_AGENT_OUTPUT_SCHEMA
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    loaded_input_example = mlflow_model.load_input_example(local_path)
    # drop the generated UUID
    loaded_input_example["messages"] = [
        {k: v for k, v in msg.items() if k != "id"} for msg in loaded_input_example["messages"]
    ]
    assert loaded_input_example == input_example

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    model_response = json.loads(response.content)
    assert model_response["messages"][0]["content"] == "You are in helpful assistant!"


def test_chat_agent_logs_default_metadata_task():
    model = SimpleChatAgent()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=model)
    assert model_info.signature.inputs == CHAT_AGENT_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_AGENT_OUTPUT_SCHEMA
    assert model_info.metadata["task"] == "agent/v2/chat"

    with mlflow.start_run():
        model_info_with_override = mlflow.pyfunc.log_model(
            name="model", python_model=model, metadata={"task": None}
        )
    assert model_info_with_override.metadata["task"] is None


def test_chat_agent_works_with_chat_agent_request_input_example():
    model = SimpleChatAgent()
    input_example_no_params = {"messages": [{"role": "user", "content": "What is rag?"}]}
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=model, input_example=input_example_no_params
        )
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == input_example_no_params

    input_example_with_params = {
        "messages": [{"role": "user", "content": "What is rag?"}],
        "context": {"conversation_id": "121", "user_id": "123"},
    }
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=model, input_example=input_example_with_params
        )
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == input_example_with_params

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    model_response = json.loads(response.content)
    assert model_response["messages"][0]["content"] == "What is rag?"


def test_chat_agent_predict_stream(tmp_path):
    model = SimpleChatAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    responses = list(loaded_model.predict_stream({"messages": messages}))
    for i, resp in enumerate(responses[:-1]):
        assert resp["delta"]["content"] == f"message {i}"


def test_chat_agent_can_receive_and_return_custom():
    messages = [{"role": "user", "content": "Hello!"}]
    input_example = {
        "messages": messages,
        "custom_inputs": {"image_url": "example", "detail": "high", "other_dict": {"key": "value"}},
    }

    model = ChatAgentWithCustomInputs()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            input_example=input_example,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # test that it works for normal pyfunc predict
    response = loaded_model.predict(input_example)
    assert response["custom_outputs"] == input_example["custom_inputs"]

    # test that it works in serving
    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    serving_response = json.loads(response.content)
    assert serving_response["custom_outputs"] == input_example["custom_inputs"]


def test_chat_agent_predict_wrapper():
    model = ChatAgentWithCustomInputs()
    dict_input_example = {
        "messages": [{"role": "user", "content": "What is rag?"}],
        "context": {"conversation_id": "121", "user_id": "123"},
        "custom_inputs": {"image_url": "example", "detail": "high", "other_dict": {"key": "value"}},
    }
    chat_agent_request = ChatAgentRequest(**dict_input_example)
    pydantic_input_example = (
        chat_agent_request.messages,
        chat_agent_request.context,
        chat_agent_request.custom_inputs,
    )
    dict_input_response = model.predict(dict_input_example)
    pydantic_input_response = model.predict(*pydantic_input_example)
    assert dict_input_response.messages[0].id is not None
    del dict_input_response.messages[0].id
    assert pydantic_input_response.messages[0].id is not None
    del pydantic_input_response.messages[0].id
    assert dict_input_response == pydantic_input_response
    no_context_dict_input_example = {**dict_input_example, "context": None}
    no_context_pydantic_input_example = (
        chat_agent_request.messages,
        None,
        chat_agent_request.custom_inputs,
    )
    dict_input_response = model.predict(no_context_dict_input_example)
    pydantic_input_response = model.predict(*no_context_pydantic_input_example)
    assert dict_input_response.messages[0].id is not None
    del dict_input_response.messages[0].id
    assert pydantic_input_response.messages[0].id is not None
    del pydantic_input_response.messages[0].id
    assert dict_input_response == pydantic_input_response

    model = SimpleChatAgent()
    dict_input_response = model.predict(dict_input_example)
    pydantic_input_response = model.predict(*pydantic_input_example)
    assert dict_input_response.messages[0].id is not None
    del dict_input_response.messages[0].id
    assert pydantic_input_response.messages[0].id is not None
    del pydantic_input_response.messages[0].id
    assert dict_input_response == pydantic_input_response
    assert list(model.predict_stream(dict_input_example)) == list(
        model.predict_stream(*pydantic_input_example)
    )

    with pytest.raises(MlflowException, match="Invalid dictionary input for a ChatAgent"):
        model.predict({"malformed dict": "bad"})
    with pytest.raises(MlflowException, match="Invalid dictionary input for a ChatAgent"):
        model.predict_stream({"malformed dict": "bad"})

    model = SimpleBadChatAgent()
    with pytest.raises(pydantic.ValidationError, match="validation error for ChatAgentResponse"):
        model.predict(dict_input_example)
    with pytest.raises(pydantic.ValidationError, match="validation error for ChatAgentChunk"):
        list(model.predict_stream(dict_input_example))


def test_chat_agent_predict_with_params(tmp_path):
    # test to codify having params in the signature
    # needed because `load_model_and_predict` in `utils/_capture_modules.py` expects a params field
    model = SimpleChatAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatAgentPyfuncWrapper)
    response = loaded_model.predict(CHAT_AGENT_INPUT_EXAMPLE, params=None)
    assert response["messages"][0]["content"] == "Hello!"

    responses = list(loaded_model.predict_stream(CHAT_AGENT_INPUT_EXAMPLE, params=None))
    for i, resp in enumerate(responses[:-1]):
        assert resp["delta"]["content"] == f"message {i}"
