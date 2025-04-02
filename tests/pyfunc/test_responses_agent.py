from uuid import uuid4

import mlflow
from mlflow.pyfunc.loaders.responses_agent import _ResponsesAgentPyfuncWrapper
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    RESPONSES_AGENT_INPUT_EXAMPLE,
    RESPONSES_AGENT_INPUT_SCHEMA,
    RESPONSES_AGENT_OUTPUT_SCHEMA,
    ResponsesRequest,
    ResponsesResponse,
)


def get_mock_response(request: ResponsesRequest, message=None):
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
    @mlflow.trace
    def predict(self, request: ResponsesRequest) -> ResponsesResponse:
        mock_response = get_mock_response(request)
        return ResponsesResponse(**mock_response)

    def predict_stream(self, request: ResponsesRequest):
        yield from get_stream_mock_response()


def test_responses_agent_save_load(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ResponsesAgentPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == RESPONSES_AGENT_INPUT_SCHEMA
    assert output_schema == RESPONSES_AGENT_OUTPUT_SCHEMA


def test_responses_agent_predict_stream(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    responses = list(loaded_model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE))
    # most of this test is that the predict_stream parsing works in _ResponsesAgentPyfuncWrapper
    for r in responses:
        assert "type" in r


def test_responses_agent_predict(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    response = loaded_model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["output"][0]["content"][0]["text"] == "Hello!"
