from typing import Generator

import pytest

from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER

if not IS_PYDANTIC_V2_OR_NEWER:
    pytest.skip(
        "ResponsesAgent and its pydantic classes are not supported in pydantic v1. Skipping test.",
        allow_module_level=True,
    )

from uuid import uuid4

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc.loaders.responses_agent import _ResponsesAgentPyfuncWrapper
from mlflow.pyfunc.model import _DEFAULT_RESPONSES_AGENT_METADATA_TASK, ResponsesAgent
from mlflow.types.responses import (
    RESPONSES_AGENT_INPUT_EXAMPLE,
    RESPONSES_AGENT_INPUT_SCHEMA,
    RESPONSES_AGENT_OUTPUT_SCHEMA,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)
from mlflow.types.schema import ColSpec, DataType, Schema


def get_mock_response(request: ResponsesRequest):
    return {
        "output": [
            {
                "type": "message",
                "id": str(uuid4()),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": request.input[0].content,
                    }
                ],
            }
        ],
    }


def get_stream_mock_response():
    yield from [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "1",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "item_id": "1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Deb",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "1",
            "output_index": 0,
            "content_index": 0,
            "delta": "rid",
        },
        {
            "type": "response.output_text.done",
            "item_id": "1",
            "output_index": 0,
            "content_index": 0,
            "text": "Debrid",
        },
        {
            "type": "response.content_part.done",
            "item_id": "1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "Debrid",
                "annotations": [],
            },
        },
    ]


class SimpleResponsesAgent(ResponsesAgent):
    def predict(self, request: ResponsesRequest) -> ResponsesResponse:
        mock_response = get_mock_response(request)
        return ResponsesResponse(**mock_response)

    def predict_stream(
        self, request: ResponsesRequest
    ) -> Generator[ResponsesStreamEvent, None, None]:
        yield from [ResponsesStreamEvent(**r) for r in get_stream_mock_response()]


def test_responses_agent_save_load_signatures(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ResponsesAgentPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == RESPONSES_AGENT_INPUT_SCHEMA
    assert output_schema == RESPONSES_AGENT_OUTPUT_SCHEMA


def test_responses_agent_log_default_task():
    model = SimpleResponsesAgent()
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model("model", python_model=model)
    assert model_info.metadata["task"] == _DEFAULT_RESPONSES_AGENT_METADATA_TASK

    with mlflow.start_run():
        model_info_with_override = mlflow.pyfunc.log_model(
            "model", python_model=model, metadata={"task": None}
        )
    assert model_info_with_override.metadata["task"] is None


def test_responses_agent_predict(tmp_path):
    model = SimpleResponsesAgent()
    response = model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response.output[0].content[0]["type"] == "output_text"
    response = model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert next(response).type == "response.output_item.added"
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    response = loaded_model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["output"][0]["content"][0]["text"] == "Hello!"


def test_responses_agent_predict_stream(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    responses = list(loaded_model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE))
    # most of this test is that the predict_stream parsing works in _ResponsesAgentPyfuncWrapper
    for r in responses:
        assert "type" in r


def test_responses_agent_with_pydantic_input():
    model = SimpleResponsesAgent()
    response = model.predict(ResponsesRequest(**RESPONSES_AGENT_INPUT_EXAMPLE))
    assert response.output[0].content[0]["text"] == "Hello!"


class CustomInputsResponsesAgent(ResponsesAgent):
    def predict(self, request: ResponsesRequest) -> ResponsesResponse:
        mock_response = get_mock_response(request)
        return ResponsesResponse(**mock_response, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesRequest):
        for r in get_stream_mock_response():
            r["custom_outputs"] = request.custom_inputs
            yield r


def test_responses_agent_custom_inputs(tmp_path):
    model = CustomInputsResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    payload = {**RESPONSES_AGENT_INPUT_EXAMPLE, "custom_inputs": {"asdf": "asdf"}}
    response = loaded_model.predict(payload)
    assert response["custom_outputs"] == {"asdf": "asdf"}
    responses = list(
        loaded_model.predict_stream(
            {**RESPONSES_AGENT_INPUT_EXAMPLE, "custom_inputs": {"asdf": "asdf"}}
        )
    )
    for r in responses:
        assert r["custom_outputs"] == {"asdf": "asdf"}


def test_responses_agent_predict_with_params(tmp_path):
    # needed because `load_model_and_predict` in `utils/_capture_modules.py` expects a params field
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    response = loaded_model.predict(RESPONSES_AGENT_INPUT_EXAMPLE, params=None)
    assert response["output"][0]["type"] == "message"


def test_responses_agent_save_throws_with_signature(tmp_path):
    model = SimpleResponsesAgent()

    with pytest.raises(MlflowException, match="Please remove the `signature` parameter"):
        mlflow.pyfunc.save_model(
            python_model=model,
            path=tmp_path,
            signature=ModelSignature(
                inputs=Schema([ColSpec(name="test", type=DataType.string)]),
            ),
        )


def test_responses_agent_throws_with_invalid_output(tmp_path):
    class BadResponsesAgent(ResponsesAgent):
        def predict(self, request: ResponsesRequest) -> ResponsesResponse:
            return {"output": [{"type": "message", "content": [{"type": "output_text"}]}]}

        def predict_stream(self, request: ResponsesRequest):
            pass

    model = BadResponsesAgent()
    with pytest.raises(
        MlflowException, match="Failed to save ResponsesAgent. Ensure your model's predict"
    ):
        mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
