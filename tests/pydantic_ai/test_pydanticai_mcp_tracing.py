import os
from unittest.mock import patch

os.environ["OPENAI_API_KEY"] = "test"
# We need to pass the api keys as environment variables as Pydantic Agent takes them from
# env variables and we can't pass them directly in the Agents
os.environ["GEMINI_API_KEY"] = "test"

import pytest
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

import mlflow
from mlflow.entities.trace import SpanType

from tests.tracing.helper import get_traces


@pytest.fixture(autouse=True)
def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.mark.asyncio
async def test_mcp_server_list_tools_autolog():
    tools_list = [
        {"name": "tool1", "description": "Tool 1 description"},
        {"name": "tool2", "description": "Tool 2 description"},
    ]

    async def list_tools(self, *args, **kwargs):
        return tools_list

    with patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools):
        mlflow.pydantic_ai.autolog(log_traces=True)

        server = MCPServerStdio(
            "deno",
            args=[
                "run",
                "-N",
                "-R=node_modules",
                "-W=node_modules",
                "--node-modules-dir=auto",
                "jsr:@pydantic/mcp-run-python",
                "stdio",
            ],
        )

        result = await server.list_tools()
        assert result == tools_list

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "MCPServerStdio.list_tools"
    assert span.span_type == SpanType.TOOL

    outputs = span.outputs
    assert len(outputs) == 2
    assert outputs == tools_list

    with patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools):
        mlflow.pydantic_ai.autolog(disable=True)
        await server.list_tools()

    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_mcp_server_call_tool_autolog():
    tool_name = "calculator"
    tool_args = {"operation": "add", "a": 5, "b": 7}
    tool_result = {"result": 12}

    async def call_tool(self, name, args, *remaining_args, **kwargs):
        assert name == tool_name
        assert args == tool_args
        return tool_result

    with patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool):
        mlflow.pydantic_ai.autolog(log_traces=True)

        server = MCPServerStdio(
            "deno",
            args=[
                "run",
                "-N",
                "-R=node_modules",
                "-W=node_modules",
                "--node-modules-dir=auto",
                "jsr:@pydantic/mcp-run-python",
                "stdio",
            ],
        )

        result = await server.call_tool(tool_name, tool_args)

        assert result == tool_result

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1

    call_tool_span = spans[0]
    assert call_tool_span is not None
    assert call_tool_span.name == "MCPServerStdio.call_tool"
    assert call_tool_span.span_type == SpanType.TOOL

    inputs = call_tool_span.inputs
    assert len(inputs) == 2
    assert inputs["name"] == tool_name
    assert inputs["args"] == tool_args

    outputs = call_tool_span.outputs
    assert len(outputs) == 1
    assert outputs == tool_result

    with patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool):
        mlflow.pydantic_ai.autolog(disable=True)
        await server.call_tool(tool_name, tool_args)

    assert len(get_traces()) == 1

# TODO: Change deno to use python based server
# @pytest.mark.asyncio
# async def test_date_calculation_with_python_code_tool():
#     CODE = """from datetime import date
#     d1 = date(2000, 1, 1)
#     d2 = date(2025, 3, 18)

#     delta = d2 - d1
#     print(delta.days)
#     """

#     USER_PROMPT = "How many days between 2000-01-01 and 2025-03-18?"

#     EXPECTED_OUTPUT = "There are 9,208 days between January 1, 2000, and March 18, 2025."

#     TOOL_NAME = "run_python_code"

#     TOOL_CALL_SEQUENCE = [
#         (
#             ModelResponse(parts=[ToolCallPart(tool_name="run_python_code", args={"code": CODE})]),
#             Usage(requests=0, request_tokens=50, response_tokens=60, total_tokens=110),
#         ),
#         (
#             ModelResponse(parts=[TextPart(content=EXPECTED_OUTPUT)]),
#             Usage(requests=1, request_tokens=100, response_tokens=150, total_tokens=250),
#         ),
#     ]

#     REQUEST_SEQUENCE = iter(TOOL_CALL_SEQUENCE)
#     TOOL_RESULT = {"output": "Days between: 9208"}
#     TOOLS_LIST = [
#         ToolDefinition(
#             name="run_python_code",
#             description="Runs Python code to perform calculations",
#             parameters_json_schema={},
#         )
#     ]

#     async def request(self, *args, **kwargs):
#         return next(REQUEST_SEQUENCE)

#     async def call_tool(self, name, args, *remaining_args, **kwargs):
#         assert name == TOOL_NAME
#         assert "code" in args
#         assert "datetime" in args["code"]
#         return TOOL_RESULT

#     async def list_tools(self, *args, **kwargs):
#         return TOOLS_LIST

#     with (
#         patch.object(InstrumentedModel, "request", new=request),
#         patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool),
#         patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools),
#     ):
#         mlflow.pydantic_ai.autolog(log_traces=True)

#         server = MCPServerStdio(
#             "deno",
#             args=[
#                 "run",
#                 "-N",
#                 "-R=node_modules",
#                 "-W=node_modules",
#                 "--node-modules-dir=auto",
#                 "jsr:@pydantic/mcp-run-python",
#                 "stdio",
#             ],
#         )

#         agent = Agent(
#             "google-gla:gemini-2.0-flash",
#             mcp_servers=[server],
#             instrument=True,
#         )

#         async with agent.run_mcp_servers():
#             result = await agent.run(USER_PROMPT)

#         assert result.output == EXPECTED_OUTPUT

#     traces = get_traces()
#     assert len(traces) == 1
#     spans = traces[0].data.spans
#     assert len(spans) == 8

#     span = spans[0]
#     assert span.name == "Agent.run"
#     assert span.span_type == SpanType.AGENT
#     assert span.inputs["user_prompt"] == USER_PROMPT
#     assert span.outputs["output"] == EXPECTED_OUTPUT

#     span1 = spans[1]
#     assert span1.name == "MCPServerStdio.list_tools_1"
#     assert span1.span_type == SpanType.TOOL
#     assert span1.parent_id == span.span_id
#     assert span1.inputs == {}
#     assert span1.outputs[0] == {
#         "name": "run_python_code",
#         "description": "Runs Python code to perform calculations",
#         "parameters_json_schema": {},
#         "outer_typed_dict_key": None,
#         "strict": None,
#     }

#     span2 = spans[2]
#     assert span2.name == "InstrumentedModel.request_1"
#     assert span2.span_type == SpanType.LLM
#     assert span2.parent_id == span.span_id
#     assert (
#         span2.inputs["args"][0][0]["parts"][0]["content"]
#         == "How many days between 2000-01-01 and 2025-03-18?"
#     )
#     assert span2.inputs["args"][2]["function_tools"][0] == {
#         "name": TOOL_NAME,
#         "description": "Runs Python code to perform calculations",
#         "parameters_json_schema": {},
#         "outer_typed_dict_key": None,
#         "strict": None,
#     }

#     span3 = spans[3]
#     assert span3.name == "MCPServerStdio.list_tools_2"
#     assert span3.span_type == SpanType.TOOL
#     assert span3.parent_id == span.span_id
#     assert span3.inputs == {}
#     assert span3.outputs[0] == {
#         "name": "run_python_code",
#         "description": "Runs Python code to perform calculations",
#         "parameters_json_schema": {},
#         "outer_typed_dict_key": None,
#         "strict": None,
#     }

#     span4 = spans[4]
#     assert span4.name == "Tool.run"
#     assert span4.span_type == SpanType.TOOL
#     assert span4.parent_id == span.span_id
#     assert span4.inputs["message"]["tool_name"] == TOOL_NAME
#     assert span4.inputs["message"]["args"]["code"] == CODE
#     assert span4.inputs["message"]["part_kind"] == "tool-call"
#     assert span4.outputs["tool_name"] == TOOL_NAME
#     assert span4.outputs["content"] == {"output": "Days between: 9208"}
#     assert span4.outputs["part_kind"] == "tool-return"

#     span5 = spans[5]
#     assert span5.name == "MCPServerStdio.call_tool"
#     assert span5.span_type == SpanType.TOOL
#     assert span5.parent_id == span4.span_id
#     assert span5.inputs == {"name": TOOL_NAME, "args": {"code": CODE}}
#     assert span5.outputs == {"output": "Days between: 9208"}

#     span6 = spans[6]
#     assert span6.name == "MCPServerStdio.list_tools_3"
#     assert span6.span_type == SpanType.TOOL
#     assert span6.parent_id == span.span_id
#     assert span6.inputs == {}
#     assert span6.outputs[0] == {
#         "name": "run_python_code",
#         "description": "Runs Python code to perform calculations",
#         "parameters_json_schema": {},
#         "outer_typed_dict_key": None,
#         "strict": None,
#     }

#     span7 = spans[7]
#     assert span7.name == "InstrumentedModel.request_2"
#     assert span7.span_type == SpanType.LLM
#     assert span7.parent_id == span.span_id

#     resp_dict, usage = span7.outputs
#     assert isinstance(usage, dict)
#     assert resp_dict["parts"][0] == {
#         "content": "There are 9,208 days between January 1, 2000, and March 18, 2025.",
#         "part_kind": "text",
#     }
#     assert usage == {
#         "requests": 1,
#         "request_tokens": 100,
#         "response_tokens": 150,
#         "total_tokens": 250,
#         "details": None,
#     }

#     with (
#         patch.object(InstrumentedModel, "request", new=request),
#         patch("pydantic_ai.mcp.MCPServer.call_tool", new=call_tool),
#         patch("pydantic_ai.mcp.MCPServer.list_tools", new=list_tools),
#     ):
#         mlflow.pydantic_ai.autolog(disable=True)

#     traces = get_traces()
#     assert len(traces) == 1
