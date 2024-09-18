import json
import pathlib
import pickle
import uuid
from dataclasses import asdict
from typing import List

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.model import Model
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import load_serving_example
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    ChatChoice,
    ChatMessage,
    ChatParams,
    ChatResponse,
    FunctionToolCallArguments,
    FunctionToolDefinition,
    ToolParamsSchema,
)
from mlflow.types.schema import ColSpec, DataType, Schema

from tests.helper_functions import (
    expect_status_code,
    pyfunc_serve_and_score_model,
)
from tests.tracing.helper import get_traces

# `None`s (`max_tokens` and `stop`) are excluded
DEFAULT_PARAMS = {
    "temperature": 1.0,
    "n": 1,
    "stream": False,
}


def get_mock_response(messages, params):
    return {
        "id": "123",
        "model": "MyChatModel",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps([m.to_dict() for m in messages]),
                },
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "role": "user",
                    "content": json.dumps(params.to_dict()),
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
    }


class TestChatModel(mlflow.pyfunc.ChatModel):
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        mock_response = get_mock_response(messages, params)
        return ChatResponse.from_dict(mock_response)


class ChatModelWithContext(mlflow.pyfunc.ChatModel):
    def load_context(self, context):
        predict_path = pathlib.Path(context.artifacts["predict_fn"])
        self.predict_fn = pickle.loads(predict_path.read_bytes())

    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        message = ChatMessage(role="assistant", content=self.predict_fn())
        return ChatResponse.from_dict(get_mock_response([message], params))


class ChatModelWithTrace(mlflow.pyfunc.ChatModel):
    @mlflow.trace
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        mock_response = get_mock_response(messages, params)
        return ChatResponse.from_dict(mock_response)


class ChatModelWithMetadata(mlflow.pyfunc.ChatModel):
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        mock_response = get_mock_response(messages, params)
        return ChatResponse(
            **mock_response,
            metadata=params.metadata,
        )


class ChatModelWithToolCalling(mlflow.pyfunc.ChatModel):
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        tools = params.tools

        # call the first tool with some value for all the required params
        tool_name = tools[0].function.name
        tool_params = tools[0].function.parameters
        arguments = {}
        for param in tool_params.required:
            param_type = tool_params.properties[param].type
            if param_type == "string":
                arguments[param] = "some_value"
            elif param_type == "number":
                arguments[param] = 123
            elif param_type == "boolean":
                arguments[param] = True
            else:
                # keep the test example simple
                raise ValueError(f"Unsupported param type: {param_type}")

        tool_call = FunctionToolCallArguments(
            name=tool_name,
            arguments=json.dumps(arguments),
        ).to_tool_call(id=uuid.uuid4().hex)

        tool_message = ChatMessage(
            role="assistant",
            tool_calls=[tool_call],
        )

        return ChatResponse(choices=[ChatChoice(index=0, message=tool_message)])


def test_chat_model_save_load(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatModelPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == CHAT_MODEL_INPUT_SCHEMA
    assert output_schema == CHAT_MODEL_OUTPUT_SCHEMA


def test_chat_model_with_trace(tmp_path):
    model = ChatModelWithTrace()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    # predict() call during saving chat model should not generate a trace
    assert len(get_traces()) == 0

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]
    loaded_model.predict({"messages": messages})

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.tags[TraceTagKey.TRACE_NAME] == "predict"
    request = json.loads(traces[0].data.request)
    assert request["messages"] == [asdict(ChatMessage.from_dict(msg)) for msg in messages]


def test_chat_model_save_throws_with_signature(tmp_path):
    model = TestChatModel()

    with pytest.raises(MlflowException, match="Please remove the `signature` parameter"):
        mlflow.pyfunc.save_model(
            python_model=model,
            path=tmp_path,
            signature=ModelSignature(
                Schema([ColSpec(name="test", type=DataType.string)]),
                Schema([ColSpec(name="test", type=DataType.string)]),
            ),
        )


def mock_predict():
    return "hello"


def test_chat_model_with_context_saves_successfully(tmp_path):
    model_path = tmp_path / "model"
    predict_path = tmp_path / "predict.pkl"
    predict_path.write_bytes(pickle.dumps(mock_predict))

    model = ChatModelWithContext()
    mlflow.pyfunc.save_model(
        python_model=model,
        path=model_path,
        artifacts={"predict_fn": str(predict_path)},
    )

    loaded_model = mlflow.pyfunc.load_model(model_path)
    messages = [{"role": "user", "content": "test"}]

    response = loaded_model.predict({"messages": messages})
    expected_response = json.dumps([{"role": "assistant", "content": "hello"}])
    assert response["choices"][0]["message"]["content"] == expected_response


@pytest.mark.parametrize(
    "ret",
    [
        "not a ChatResponse",
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
    class BadChatModel(mlflow.pyfunc.ChatModel):
        def predict(self, context, messages, params) -> ChatResponse:
            return ret

    model = BadChatModel()
    with pytest.raises(
        MlflowException,
        match=(
            "Failed to save ChatModel. Please ensure that the model's "
            r"predict\(\) method returns a ChatResponse object"
        ),
    ):
        mlflow.pyfunc.save_model(python_model=model, path=tmp_path)


# test that we can predict with the model
def test_chat_model_predict(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]

    response = loaded_model.predict({"messages": messages})
    assert response["choices"][0]["message"]["content"] == json.dumps(messages)
    assert json.loads(response["choices"][1]["message"]["content"]) == DEFAULT_PARAMS

    # override all params
    params_override = {
        "temperature": 0.5,
        "max_tokens": 10,
        "stop": ["\n"],
        "n": 2,
        "stream": True,
        "top_p": 0.1,
        "top_k": 20,
        "frequency_penalty": 0.5,
        "presence_penalty": -0.5,
    }
    response = loaded_model.predict({"messages": messages, **params_override})
    assert response["choices"][0]["message"]["content"] == json.dumps(messages)
    assert json.loads(response["choices"][1]["message"]["content"]) == params_override

    # override a subset of params
    params_subset = {
        "max_tokens": 100,
    }
    response = loaded_model.predict({"messages": messages, **params_subset})
    assert response["choices"][0]["message"]["content"] == json.dumps(messages)
    assert json.loads(response["choices"][1]["message"]["content"]) == {
        **DEFAULT_PARAMS,
        **params_subset,
    }


def test_chat_model_works_in_serving():
    model = TestChatModel()
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]
    params_subset = {
        "max_tokens": 100,
    }
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            input_example=(messages, params_subset),
        )

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    choices = json.loads(response.content)["choices"]
    assert choices[0]["message"]["content"] == json.dumps(messages)
    assert json.loads(choices[1]["message"]["content"]) == {
        **DEFAULT_PARAMS,
        **params_subset,
    }


def test_chat_model_works_with_infer_signature_input_example(tmp_path):
    model = TestChatModel()
    params_subset = {
        "max_tokens": 100,
    }
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is Retrieval-augmented Generation?",
            }
        ],
        **params_subset,
    }
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=model, input_example=input_example
        )
    assert model_info.signature.inputs == CHAT_MODEL_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_MODEL_OUTPUT_SCHEMA
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == {
        "messages": input_example["messages"],
        **DEFAULT_PARAMS,
        **params_subset,
    }

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    choices = json.loads(response.content)["choices"]
    assert choices[0]["message"]["content"] == json.dumps(input_example["messages"])
    assert json.loads(choices[1]["message"]["content"]) == {
        **DEFAULT_PARAMS,
        **params_subset,
    }


def test_chat_model_works_with_chat_message_input_example(tmp_path):
    model = TestChatModel()
    input_example = [
        ChatMessage(role="user", content="What is Retrieval-augmented Generation?", name="chat")
    ]
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=model, input_example=input_example
        )
    assert model_info.signature.inputs == CHAT_MODEL_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_MODEL_OUTPUT_SCHEMA
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == {
        "messages": [message.to_dict() for message in input_example],
        **DEFAULT_PARAMS,
    }

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    choices = json.loads(response.content)["choices"]
    assert choices[0]["message"]["content"] == json.dumps(json.loads(inference_payload)["messages"])


def test_chat_model_works_with_infer_signature_multi_input_example(tmp_path):
    model = TestChatModel()
    params_subset = {
        "max_tokens": 100,
    }
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
        **params_subset,
    }
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=model, input_example=input_example
        )
    assert model_info.signature.inputs == CHAT_MODEL_INPUT_SCHEMA
    assert model_info.signature.outputs == CHAT_MODEL_OUTPUT_SCHEMA
    mlflow_model = Model.load(model_info.model_uri)
    local_path = _download_artifact_from_uri(model_info.model_uri)
    assert mlflow_model.load_input_example(local_path) == {
        "messages": input_example["messages"],
        **DEFAULT_PARAMS,
        **params_subset,
    }

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)
    choices = json.loads(response.content)["choices"]
    assert choices[0]["message"]["content"] == json.dumps(input_example["messages"])
    assert json.loads(choices[1]["message"]["content"]) == {
        **DEFAULT_PARAMS,
        **params_subset,
    }


def test_chat_model_predict_stream(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]

    response = next(loaded_model.predict_stream({"messages": messages}))
    assert response["choices"][0]["message"]["content"] == json.dumps(messages)


def test_chat_model_can_receive_and_return_metadata():
    messages = [{"role": "user", "content": "Hello!"}]
    params = {
        "metadata": {
            "image_url": "example",
            "detail": "high",
        },
    }
    input_example = {
        "messages": messages,
        **params,
    }

    model = ChatModelWithMetadata()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            input_example=input_example,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # test that it works for normal pyfunc predict
    response = loaded_model.predict({"messages": messages, **params})
    assert response["metadata"] == params["metadata"]

    # test that it works in serving
    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    serving_response = json.loads(response.content)
    assert serving_response["metadata"] == params["metadata"]


def test_chat_model_can_use_tool_calls():
    messages = [{"role": "user", "content": "What's the weather?"}]

    weather_tool = (
        FunctionToolDefinition(
            name="get_weather",
            description="Get the weather for your current location",
            parameters=ToolParamsSchema(
                {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for",
                    },
                    "unit": {"type": "string", "enum": ["F", "C"]},
                },
                required=["city", "unit"],
            ),
        )
        .to_tool_definition()
        .to_dict()
    )

    example = {
        "messages": messages,
        "tools": [weather_tool],
    }

    model = ChatModelWithToolCalling()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            input_example=example,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_model.predict(example)

    model_tool_calls = response["choices"][0]["message"]["tool_calls"]
    assert json.loads(model_tool_calls[0]["function"]["arguments"]) == {
        "city": "some_value",
        "unit": "some_value",
    }
