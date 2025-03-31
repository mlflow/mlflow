from uuid import uuid4

import mlflow
from mlflow.pyfunc.loaders.responses_agent import _ResponsesAgentPyfuncWrapper
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    RESPONSES_AGENT_INPUT_SCHEMA,
    RESPONSES_AGENT_OUTPUT_SCHEMA,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
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
                        "text": message or msg.content,
                    }
                    for msg in request.input
                ],
            }
            for msg in request.input
        ],
    }


class SimpleResponsesAgent(ResponsesAgent):
    @mlflow.trace
    def predict(self, request: ResponsesRequest) -> ResponsesResponse:
        mock_response = get_mock_response(request)
        return ResponsesResponse(**mock_response)

    def predict_stream(self, request: ResponsesRequest):
        for i in range(5):
            mock_response = get_mock_response(request, f"message {i}")
            mock_response["delta"] = mock_response["messages"][0]
            mock_response["delta"]["id"] = str(i)
            yield ResponsesStreamEvent(**mock_response)


def test_responses_agent_save_load(tmp_path):
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=tmp_path)

    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert isinstance(loaded_model._model_impl, _ResponsesAgentPyfuncWrapper)
    input_schema = loaded_model.metadata.get_input_schema()
    output_schema = loaded_model.metadata.get_output_schema()
    assert input_schema == RESPONSES_AGENT_INPUT_SCHEMA
    assert output_schema == RESPONSES_AGENT_OUTPUT_SCHEMA
