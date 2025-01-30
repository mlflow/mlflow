import json

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
    CHAT_AGENT_INPUT_SCHEMA,
    CHAT_AGENT_OUTPUT_SCHEMA,
    ChatAgentMessage,
    ChatAgentRequest,
    ChatAgentResponse,
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
                "id": "call_4a7173b9-5058-4236-adf0-d3fdde69076c",
            }
            for msg in messages
        ],
    }


class SimpleChatAgent(ChatAgent):
    @mlflow.trace
    def predict(self, model_input: ChatAgentRequest) -> ChatAgentResponse:
        mock_response = get_mock_response(model_input.messages)
        return ChatAgentResponse(**mock_response)

    def predict_stream(self, model_input: ChatAgentRequest):
        for i in range(5):
            mock_response = get_mock_response(model_input.messages, f"message {i}")
            yield ChatAgentResponse(**mock_response)


class SimpleDictChatAgent(ChatAgent):
    @mlflow.trace
    def predict(self, model_input: ChatAgentRequest) -> ChatAgentResponse:
        mock_response = get_mock_response(model_input.messages)
        return ChatAgentResponse(**mock_response).model_dump_compat()


class ChatAgentWithCustomInputs(ChatAgent):
    def predict(self, model_input: ChatAgentRequest) -> ChatAgentResponse:
        mock_response = get_mock_response(model_input.messages)
        return ChatAgentResponse(
            **mock_response,
            custom_outputs=model_input.custom_inputs,
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
    assert [
        {k: v for k, v in msg.items() if k != "id"} for msg in request["model_input"]["messages"]
    ] == [
        {k: v for k, v in ChatAgentMessage(**msg).model_dump_compat().items() if k != "id"}
        for msg in messages
    ]
    assert False


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


def mock_predict():
    return "hello"


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
        def predict(self, model_input) -> ChatAgentResponse:
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
                "role": "assistant",
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
            "model", python_model=model, input_example=input_example
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
        model_info = mlflow.pyfunc.log_model("model", python_model=model)
    assert model_info.signature.inputs == CHAT_AGENT_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_AGENT_OUTPUT_SCHEMA
    assert model_info.metadata["task"] == "agent/v2/chat"

    with mlflow.start_run():
        model_info_with_override = mlflow.pyfunc.log_model(
            "model", python_model=model, metadata={"task": None}
        )
    assert model_info_with_override.metadata["task"] is None


def test_chat_agent_works_with_chat_agent_request_input_example():
    model = SimpleChatAgent()
    input_example_no_params = ChatAgentRequest(
        **{"messages": [ChatAgentMessage(role="user", content="What is rag?")]}
    )
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=model, input_example=input_example_no_params
        )
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == input_example_no_params.model_dump_compat(
        exclude_none=True
    )

    input_example_with_params = ChatAgentRequest(
        messages=[ChatAgentMessage(role="user", content="What is Retrieval-augmented Generation?")],
        context={"conversation_id": "121", "user_id": "123"},
    )
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=model, input_example=input_example_with_params
        )
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(
        local_path
    ) == input_example_with_params.model_dump_compat(exclude_none=True)

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
        assert resp["messages"][0]["content"] == f"message {i}"


def test_chat_agent_can_receive_and_return_custom():
    messages = [{"role": "user", "content": "Hello!"}]
    params = {
        "custom_inputs": {"image_url": "example", "detail": "high", "other_dict": {"key": "value"}},
    }
    input_example = {
        "messages": messages,
        **params,
    }

    model = ChatAgentWithCustomInputs()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            input_example=input_example,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # test that it works for normal pyfunc predict
    response = loaded_model.predict({"messages": messages, **params})
    assert response["custom_outputs"] == params["custom_inputs"]

    # test that it works in serving
    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    serving_response = json.loads(response.content)
    assert serving_response["custom_outputs"] == params["custom_inputs"]
