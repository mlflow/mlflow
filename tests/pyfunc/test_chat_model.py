import json
from typing import List

import mlflow
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    ChatMessage,
    ChatParams,
    ChatResponse,
)

from tests.helper_functions import expect_status_code, pyfunc_serve_and_score_model

# `None`s (`max_tokens` and `stop`) are excluded
DEFAULT_PARAMS = {
    "temperature": 1.0,
    "n": 1,
    "stream": False,
}


class TestChatModel(mlflow.pyfunc.ChatModel):
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        mock_response = {
            "id": "123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "MyChatModel",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps([m.model_dump(exclude_none=True) for m in messages]),
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "user",
                        "content": params.model_dump_json(exclude_none=True),
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
        return ChatResponse(**mock_response)


def test_chat_model_save_load(tmp_path):
    model = TestChatModel()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ChatModelPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == CHAT_MODEL_INPUT_SCHEMA
    assert output_schema == CHAT_MODEL_OUTPUT_SCHEMA


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
