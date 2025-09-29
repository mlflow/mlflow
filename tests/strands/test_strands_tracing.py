import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from strands import Agent
from strands.models.model import Model
from strands.tools.tools import PythonAgentTool

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


async def sum_tool(tool_use, **_):
    a = tool_use["input"]["a"]
    b = tool_use["input"]["b"]
    return {
        "toolUseId": tool_use["toolUseId"],
        "status": "success",
        "content": [{"json": a + b}],
    }


tool = PythonAgentTool(
    "sum",
    {
        "name": "sum",
        "description": "add numbers 1 2",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    sum_tool,
)


class DummyModel(Model):
    def __init__(self, response_text: str, in_tokens: int = 1, out_tokens: int = 1):
        self.response_text = response_text
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.config = {}

    def update_config(self, **model_config):
        self.config.update(model_config)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:
            yield {}

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        tool_specs: Any | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": self.response_text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": self.in_tokens,
                    "outputTokens": self.out_tokens,
                    "totalTokens": self.in_tokens + self.out_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        }


class ToolCallingModel(Model):
    def __init__(
        self,
        response_text: str,
        tool_input: dict[str, Any] | None = None,
        tool_name: str = "sum",
    ):
        self.response_text = response_text
        self.tool_input = tool_input or {"a": 1, "b": 2}
        self.tool_name = tool_name
        self.config = {}
        self._call_count = 0

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, object]:
        return self.config

    async def structured_output(
        self,
        output_model: Any,
        prompt: Any,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        tool_specs: Any | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if self._call_count == 0:
            self._call_count += 1
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "tool-1",
                            "name": self.tool_name,
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.tool_input)}}}
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "totalTokens": 2,
                    },
                    "metrics": {"latencyMs": 0},
                }
            }
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": self.response_text}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "totalTokens": 2,
                    },
                    "metrics": {"latencyMs": 0},
                }
            }


def test_strands_autolog_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=DummyModel("hi", 1, 2), name="agent")
    agent("hello")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    agent_span = next(span for span in spans if span.span_type == SpanType.AGENT)
    assert agent_span.inputs == [{"role": "user", "content": [{"text": "hello"}]}]
    assert agent_span.outputs.strip() == "hi"
    assert SpanAttributeKey.CHAT_USAGE not in agent_span.attributes

    usage_spans = [span for span in spans if span.attributes.get(SpanAttributeKey.CHAT_USAGE)]
    assert usage_spans, "expected at least one child span recording token usage"
    assert all(span.span_type != SpanType.AGENT for span in usage_spans)
    assert usage_spans[0].attributes[SpanAttributeKey.CHAT_USAGE] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
    assert traces[0].info.token_usage == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }

    mlflow.strands.autolog(disable=True)
    agent("bye")
    assert len(get_traces()) == 1


def test_function_calling_creates_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=ToolCallingModel("3"), tools=[tool], name="agent")
    agent("add numbers 1 2 1 2")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    agent_span = spans[0]
    assert agent_span.span_type == SpanType.AGENT
    tool_span = spans[3]
    assert tool_span.span_type == SpanType.TOOL
    assert agent_span.inputs == [{"role": "user", "content": [{"text": "add numbers 1 2 1 2"}]}]
    assert agent_span.outputs == 3
    assert tool_span.inputs == [{"role": "tool", "content": {"a": 1, "b": 2}}]
    assert tool_span.outputs == [{"json": 3}]


def test_multiple_agents_single_trace():
    mlflow.strands.autolog()

    agent2 = Agent(model=DummyModel("hi"), name="agent2")

    async def sum_and_call_agent2(
        tool_use: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        a = tool_use["input"]["a"]
        b = tool_use["input"]["b"]
        await agent2.invoke_async("hello")
        return {
            "toolUseId": tool_use["toolUseId"],
            "status": "success",
            "content": [{"json": a + b}],
        }

    tool_with_agent2 = PythonAgentTool(
        "sum",
        {
            "name": "sum",
            "description": "add numbers 1 2",
            "inputSchema": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        sum_and_call_agent2,
    )

    agent1 = Agent(model=ToolCallingModel("3"), tools=[tool_with_agent2], name="agent1")
    agent1("add numbers 1 2")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    agent1_span = spans[0]
    assert agent1_span.name == "invoke_agent agent1"
    tool_span = spans[3]
    assert tool_span.span_type == SpanType.TOOL
    agent2_span = spans[4]
    assert agent2_span.name == "invoke_agent agent2"
    assert agent1_span.inputs == [{"role": "user", "content": [{"text": "add numbers 1 2"}]}]
    assert agent1_span.outputs == 3
    assert tool_span.inputs == [{"role": "tool", "content": {"a": 1, "b": 2}}]
    assert tool_span.outputs == [{"json": 3}]
    assert agent2_span.inputs == [{"role": "user", "content": [{"text": "hello"}]}]
    assert agent2_span.outputs.strip() == "hi"


def test_autolog_disable_prevents_new_traces():
    mlflow.strands.autolog()

    agent1 = Agent(model=DummyModel("hi"), name="agent1")
    agent2 = Agent(model=DummyModel("cya"), name="agent2")

    agent1("hello")
    assert len(get_traces()) == 1

    mlflow.strands.autolog(disable=True)
    agent2("bye")
    assert len(get_traces()) == 1
