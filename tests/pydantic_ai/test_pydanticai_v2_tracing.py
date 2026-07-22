import importlib.metadata

import pytest
from packaging.version import Version

if Version(importlib.metadata.version("pydantic_ai")).major < 2:
    pytest.skip("Pydantic AI 2.x tracing tests", allow_module_level=True)

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

import mlflow
from mlflow.entities import SpanType
from mlflow.pydantic_ai.autolog_v2 import (
    _StreamedRunResultSyncWrapper,
    patched_capability_tool_execute,
    patched_mcp_direct_call_tool,
    patched_mcp_list_tools,
)

from tests.tracing.helper import get_traces


def _span_names():
    traces = get_traces()
    assert len(traces) == 1
    return [span.name for span in traces[0].data.spans]


def test_autolog_enables_agent_instrumentation():
    mlflow.pydantic_ai.autolog()

    agent = Agent(TestModel())

    assert agent.instrument is True


def test_existing_explicit_instrument_false_is_respected():
    agent = Agent(TestModel(custom_output_text="hello"))
    agent.instrument = False

    mlflow.pydantic_ai.autolog()
    assert agent.run_sync("hi").output == "hello"

    assert agent.instrument is False
    assert _span_names() == ["Agent.run_sync", "Agent.run"]


def test_disable_removes_v2_patches():
    mlflow.pydantic_ai.autolog()
    mlflow.pydantic_ai.autolog(disable=True)

    agent = Agent(TestModel(custom_output_text="hello"))
    assert agent.instrument is None
    assert agent.run_sync("hi").output == "hello"
    assert get_traces() == []


def test_run_sync_preserves_run_nesting():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    assert agent.run_sync("hi").output == "hello"

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [span.name for span in spans] == [
        "Agent.run_sync",
        "Agent.run",
        "TestModel.request",
    ]
    assert [span.span_type for span in spans] == [
        SpanType.AGENT,
        SpanType.AGENT,
        SpanType.LLM,
    ]
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].parent_id == spans[1].span_id


def test_tool_span_uses_logical_tool_payload():
    mlflow.pydantic_ai.autolog()

    def add(left: int, right: int) -> int:
        return left + right

    agent = Agent(TestModel(), tools=[add])
    agent.run_sync("add two numbers")

    traces = get_traces()
    assert len(traces) == 1
    tool_span = next(span for span in traces[0].data.spans if span.name == "add")
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.inputs == {"left": 0, "right": 0}
    assert tool_span.outputs == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["validation_error", "model_retry"])
async def test_tool_validation_failure_is_traced(error_type):
    mlflow.pydantic_ai.autolog()
    instrumentation = Instrumentation()
    call = ToolCallPart(tool_name="add", args={"left": "invalid", "right": 1})
    tool_def = ToolDefinition(name="add", parameters_json_schema={})

    if error_type == "validation_error":

        class Arguments(BaseModel):
            left: int

        with pytest.raises(ValidationError, match="valid integer") as exc_info:
            Arguments.model_validate(call.args)
        error = exc_info.value
    else:
        error = ModelRetry("try different arguments")

    with pytest.raises(type(error)):
        await instrumentation.on_tool_validate_error(
            None,
            call=call,
            tool_def=tool_def,
            args=call.args,
            error=error,
        )

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.name == "add.validation"
    assert span.span_type == SpanType.PARSER
    assert span.inputs == {"left": "invalid", "right": 1}
    assert span.status.status_code == "ERROR"
    assert len(span.events) == 1


def test_validation_failure_remains_visible_after_successful_retry():
    mlflow.pydantic_ai.autolog()
    request_count = 0

    def model_function(_messages, _agent_info):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            part = ToolCallPart(tool_name="add", args={"left": "invalid", "right": 1})
        elif request_count == 2:
            part = ToolCallPart(tool_name="add", args={"left": 2, "right": 1})
        else:
            part = TextPart(content="done")
        return ModelResponse(parts=[part])

    def add(left: int, right: int) -> int:
        return left + right

    agent = Agent(FunctionModel(model_function), tools=[add])

    assert agent.run_sync("add two numbers").output == "done"

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    validation_span = next(span for span in spans if span.name == "add.validation")
    tool_span = next(span for span in spans if span.name == "add")
    assert validation_span.status.status_code == "ERROR"
    assert validation_span.inputs == {"left": "invalid", "right": 1}
    assert tool_span.status.status_code == "OK"
    assert tool_span.inputs == {"left": 2, "right": 1}
    assert tool_span.outputs == 3
    assert spans[0].status.status_code == "OK"


@pytest.mark.asyncio
async def test_mcp_list_tools_uses_public_transport_payload():
    mlflow.pydantic_ai.autolog()
    toolset = object.__new__(MCPToolset)

    async def list_tools(self):
        return [{"name": "remote_tool"}]

    result = await patched_mcp_list_tools(list_tools, toolset)

    assert result == [{"name": "remote_tool"}]
    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.name == "MCPToolset.list_tools"
    assert span.span_type == SpanType.TOOL
    assert span.inputs == {}
    assert span.outputs == [{"name": "remote_tool"}]


@pytest.mark.asyncio
async def test_mcp_tool_has_logical_and_nested_transport_spans():
    mlflow.pydantic_ai.autolog()
    instrumentation = object.__new__(Instrumentation)
    toolset = object.__new__(MCPToolset)
    call = ToolCallPart(tool_name="remote_tool", args={"value": 2})
    tool_def = ToolDefinition(name="remote_tool", parameters_json_schema={})

    async def direct_call_tool(self, name, args, *, metadata=None, use_task=False):
        assert metadata == {"request": "metadata"}
        assert use_task is True
        return {"doubled": args["value"] * 2}

    async def execute_handler(args):
        return await patched_mcp_direct_call_tool(
            direct_call_tool,
            toolset,
            "remote_tool",
            args,
            metadata={"request": "metadata"},
            use_task=True,
        )

    async def execute(self, ctx, *, call, tool_def, args, handler):
        return await handler(args)

    result = await patched_capability_tool_execute(
        execute,
        instrumentation,
        None,
        call=call,
        tool_def=tool_def,
        args={"value": 2},
        handler=execute_handler,
    )

    assert result == {"doubled": 4}
    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [span.name for span in spans] == ["remote_tool", "MCPToolset.direct_call_tool"]
    assert spans[0].inputs == {"value": 2}
    assert spans[0].outputs == {"doubled": 4}
    assert spans[1].inputs == {"name": "remote_tool", "args": {"value": 2}}
    assert spans[1].outputs == {"doubled": 4}
    assert spans[1].parent_id == spans[0].span_id


@pytest.mark.asyncio
async def test_run_is_traced():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    assert (await agent.run("hi")).output == "hello"
    assert _span_names() == ["Agent.run", "TestModel.request"]


@pytest.mark.asyncio
async def test_run_stream_is_traced():
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    async with agent.run_stream("hi") as result:
        assert await result.get_output() == "hello"
    assert _span_names() == ["Agent.run_stream", "TestModel.request"]


@pytest.mark.parametrize(
    "completion_method",
    ["stream_text", "stream_output", "stream_response", "get_output"],
)
@pytest.mark.parametrize("use_context_manager", [True, False])
def test_run_stream_sync_lifecycle(completion_method, use_context_manager):
    mlflow.pydantic_ai.autolog()
    agent = Agent(TestModel(custom_output_text="hello"))

    result = agent.run_stream_sync("hi")

    def consume():
        value = getattr(result, completion_method)()
        return value if completion_method == "get_output" else list(value)

    if use_context_manager:
        with result:
            output = consume()
    else:
        output = consume()

    assert output
    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert [span.name for span in spans] == [
        "Agent.run_stream_sync",
        "Agent.run_stream",
        "TestModel.request",
    ]
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].parent_id == spans[1].span_id


def test_run_stream_sync_cleanup_error_is_propagated_and_traced():
    class FailingResult:
        def __exit__(self, exc_type, exc_value, traceback):
            raise RuntimeError("cleanup failed")

    span = mlflow.start_span_no_context(
        name="Agent.run_stream_sync",
        span_type=SpanType.AGENT,
    )
    result = _StreamedRunResultSyncWrapper(FailingResult(), span)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        result.__exit__(None, None, None)

    traces = get_traces()
    assert len(traces) == 1
    recorded_span = traces[0].data.spans[0]
    assert recorded_span.status.status_code == "ERROR"
    assert recorded_span.events[0].attributes["exception.message"] == "cleanup failed"
