import functools
import pathlib
import pickle
from typing import Generator

import pytest

from mlflow.entities.span import SpanType
from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER

from tests.tracing.helper import get_traces, purge_traces

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
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.schema import ColSpec, DataType, Schema


def get_mock_response(request: ResponsesAgentRequest):
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
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        mock_response = get_mock_response(request)
        return ResponsesAgentResponse(**mock_response)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        yield from [ResponsesAgentStreamEvent(**r) for r in get_stream_mock_response()]


class ResponsesAgentWithContext(ResponsesAgent):
    def load_context(self, context):
        predict_path = pathlib.Path(context.artifacts["predict_fn"])
        self.predict_fn = pickle.loads(predict_path.read_bytes())

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        return ResponsesAgentResponse(
            output=[
                {
                    "type": "message",
                    "id": "test-id",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": self.predict_fn(),
                        }
                    ],
                }
            ]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        yield ResponsesAgentStreamEvent(
            type="response.output_item.added",
            output_index=0,
            item=self.create_text_output_item(self.predict_fn(), "test-id"),
        )


def mock_responses_predict():
    return "hello from context"


def test_responses_agent_with_context(tmp_path):
    predict_path = tmp_path / "predict.pkl"
    predict_path.write_bytes(pickle.dumps(mock_responses_predict))

    model = ResponsesAgentWithContext()

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            artifacts={"predict_fn": str(predict_path)},
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # Test predict
    response = loaded_model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response["output"][0]["content"][0]["text"] == "hello from context"

    # Test predict_stream
    responses = list(loaded_model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE))
    assert len(responses) == 1
    assert responses[0]["item"]["content"][0]["text"] == "hello from context"


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
        model_info = mlflow.pyfunc.log_model(name="model", python_model=model)
    assert model_info.metadata["task"] == _DEFAULT_RESPONSES_AGENT_METADATA_TASK

    with mlflow.start_run():
        model_info_with_override = mlflow.pyfunc.log_model(
            name="model", python_model=model, metadata={"task": None}
        )
    assert model_info_with_override.metadata["task"] is None


def test_responses_agent_predict(tmp_path):
    model_path = tmp_path / "model"
    model = SimpleResponsesAgent()
    response = model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response.output[0].content[0]["type"] == "output_text"
    response = model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert next(response).type == "response.output_item.added"
    mlflow.pyfunc.save_model(python_model=model, path=model_path)
    loaded_model = mlflow.pyfunc.load_model(model_path)
    response = loaded_model.predict(RESPONSES_AGENT_INPUT_EXAMPLE)
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["output"][0]["content"][0]["text"] == "Hello!"


def test_responses_agent_predict_stream(tmp_path):
    model_path = tmp_path / "model"
    model = SimpleResponsesAgent()
    mlflow.pyfunc.save_model(python_model=model, path=model_path)
    loaded_model = mlflow.pyfunc.load_model(model_path)
    responses = list(loaded_model.predict_stream(RESPONSES_AGENT_INPUT_EXAMPLE))
    # most of this test is that the predict_stream parsing works in _ResponsesAgentPyfuncWrapper
    for r in responses:
        assert "type" in r


def test_responses_agent_with_pydantic_input():
    model = SimpleResponsesAgent()
    response = model.predict(ResponsesAgentRequest(**RESPONSES_AGENT_INPUT_EXAMPLE))
    assert response.output[0].content[0]["text"] == "Hello!"


class CustomInputsResponsesAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        mock_response = get_mock_response(request)
        return ResponsesAgentResponse(**mock_response, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest):
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
        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            return {"output": [{"type": "message", "content": [{"type": "output_text"}]}]}

    model = BadResponsesAgent()
    with pytest.raises(
        MlflowException, match="Failed to save ResponsesAgent. Ensure your model's predict"
    ):
        mlflow.pyfunc.save_model(python_model=model, path=tmp_path)


@pytest.mark.parametrize(
    ("input", "outputs"),
    [
        # 1. Normal text input output
        (
            RESPONSES_AGENT_INPUT_EXAMPLE,
            {
                "output": [
                    {
                        "type": "message",
                        "id": "test",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Dummy output"}],
                    }
                ],
            },
        ),
        # 2. Image input
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "what is in this image?"},
                            {"type": "input_image", "image_url": "test.jpg"},
                        ],
                    }
                ],
            },
            {
                "output": [
                    {
                        "type": "message",
                        "id": "test",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Dummy output"}],
                    }
                ],
            },
        ),
        # 3. Tool calling
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": "What is the weather like in Boston today?",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_current_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location", "unit"],
                        },
                    }
                ],
            },
            {
                "output": [
                    {
                        "arguments": '{"location":"Boston, MA","unit":"celsius"}',
                        "call_id": "function_call_1",
                        "name": "get_current_weather",
                        "type": "function_call",
                        "id": "fc_6805c835567481918c27724bbe931dc40b1b7951a48825bb",
                        "status": "completed",
                    }
                ]
            },
        ),
    ],
)
def test_responses_agent_trace(input, outputs):
    class TracedResponsesAgent(ResponsesAgent):
        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            return ResponsesAgentResponse(**outputs)

        def predict_stream(
            self, request: ResponsesAgentRequest
        ) -> Generator[ResponsesAgentStreamEvent, None, None]:
            for item in outputs["output"]:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=item,
                )

    model = TracedResponsesAgent()
    model.predict(ResponsesAgentRequest(**input))

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "predict"
    assert spans[0].span_type == SpanType.AGENT

    list(model.predict_stream(ResponsesAgentRequest(**input)))

    traces = get_traces()
    assert len(traces) == 2
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "predict_stream"
    assert spans[0].span_type == SpanType.AGENT

    assert "output" in spans[0].outputs
    assert spans[0].outputs["output"] == outputs["output"]


def test_responses_agent_custom_trace_configurations():
    """Test that custom trace configurations are preserved when functions are already traced."""

    # Agent with custom span names and attributes
    class CustomTracedAgent(ResponsesAgent):
        @mlflow.trace(
            name="custom_predict", span_type=SpanType.AGENT, attributes={"custom": "value"}
        )
        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            return ResponsesAgentResponse(**get_mock_response(request))

        @mlflow.trace(
            name="custom_predict_stream",
            span_type=SpanType.AGENT,
            attributes={"stream": "true"},
            output_reducer=ResponsesAgent.responses_agent_output_reducer,
        )
        def predict_stream(
            self, request: ResponsesAgentRequest
        ) -> Generator[ResponsesAgentStreamEvent, None, None]:
            yield from [ResponsesAgentStreamEvent(**r) for r in get_stream_mock_response()]

    purge_traces()

    agent = CustomTracedAgent()
    agent.predict(ResponsesAgentRequest(**RESPONSES_AGENT_INPUT_EXAMPLE))

    traces_predict = get_traces()
    assert len(traces_predict) == 1
    spans_predict = traces_predict[0].data.spans
    assert len(spans_predict) == 1
    assert spans_predict[0].name == "custom_predict"
    assert spans_predict[0].span_type == SpanType.AGENT
    assert spans_predict[0].attributes.get("custom") == "value"

    purge_traces()
    list(agent.predict_stream(ResponsesAgentRequest(**RESPONSES_AGENT_INPUT_EXAMPLE)))

    traces_stream = get_traces()
    assert len(traces_stream) == 1
    spans_stream = traces_stream[0].data.spans
    assert len(spans_stream) == 1
    assert spans_stream[0].name == "custom_predict_stream"
    assert spans_stream[0].span_type == SpanType.AGENT
    assert spans_stream[0].attributes.get("stream") == "true"


def test_responses_agent_non_mlflow_decorators():
    """Test that the implementation correctly distinguishes MLflow tracing from other decorators."""

    # Create a custom decorator to test with
    def custom_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    class MixedDecoratedAgent(ResponsesAgent):
        @custom_decorator
        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            return ResponsesAgentResponse(**get_mock_response(request))

        # Just a regular method (no decorator) to test that it gets auto-traced
        def predict_stream(
            self, request: ResponsesAgentRequest
        ) -> Generator[ResponsesAgentStreamEvent, None, None]:
            yield from [ResponsesAgentStreamEvent(**r) for r in get_stream_mock_response()]

    # Both methods should get auto-traced since they don't have __mlflow_traced__
    agent = MixedDecoratedAgent()
    agent.predict(ResponsesAgentRequest(**RESPONSES_AGENT_INPUT_EXAMPLE))

    traces_mixed_predict = get_traces()
    assert len(traces_mixed_predict) == 1
    spans_mixed_predict = traces_mixed_predict[0].data.spans
    assert len(spans_mixed_predict) == 1
    assert spans_mixed_predict[0].name == "predict"
    assert spans_mixed_predict[0].span_type == SpanType.AGENT

    purge_traces()
    list(agent.predict_stream(ResponsesAgentRequest(**RESPONSES_AGENT_INPUT_EXAMPLE)))

    traces_mixed_stream = get_traces()
    assert len(traces_mixed_stream) == 1
    spans_mixed_stream = traces_mixed_stream[0].data.spans
    assert len(spans_mixed_stream) == 1
    assert spans_mixed_stream[0].name == "predict_stream"
    assert spans_mixed_stream[0].span_type == SpanType.AGENT
