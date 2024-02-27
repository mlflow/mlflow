import json
import pathlib
import pickle
from typing import List

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    ChatMessage,
    ChatParams,
    ChatResponse,
)
from mlflow.types.schema import ColSpec, DataType, Schema

from tests.helper_functions import expect_status_code, pyfunc_serve_and_score_model

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
        return ChatResponse(**mock_response)


class ChatModelWithContext(mlflow.pyfunc.ChatModel):
    def load_context(self, context):
        predict_path = pathlib.Path(context.artifacts["predict_fn"])
        self.predict_fn = pickle.loads(predict_path.read_bytes())

    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        message = ChatMessage(role="assistant", content=self.predict_fn())
        return ChatResponse(**get_mock_response([message], params))


def test_chat_model_save_load(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatModelPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == CHAT_MODEL_INPUT_SCHEMA
    assert output_schema == CHAT_MODEL_OUTPUT_SCHEMA


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


def test_chat_model_works_in_serving(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]
    params_subset = {
        "max_tokens": 100,
    }

    response = pyfunc_serve_and_score_model(
        model_uri=tmp_path,
        data=json.dumps({"messages": messages, **params_subset}),
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
