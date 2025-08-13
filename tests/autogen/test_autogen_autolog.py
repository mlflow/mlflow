import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import FunctionCall, Image
from autogen_core.models import CreateResult
from autogen_ext.models.replay import ReplayChatCompletionClient

import mlflow
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

_SYSTEM_MESSAGE = "You are a helpful assistant."
_MODEL_USAGE = {"prompt_tokens": 6, "completion_tokens": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disable",
    [True, False],
)
async def test_autolog_assistant_agent(disable):
    model_client = ReplayChatCompletionClient(
        ["2"],
    )
    agent = AssistantAgent("assistant", model_client=model_client, system_message=_SYSTEM_MESSAGE)

    mlflow.autogen.autolog(disable=disable)

    await agent.run(task="1+1")

    traces = get_traces()

    if disable:
        assert len(traces) == 0
    else:
        assert len(traces) == 1
        trace = traces[0]
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 3
        span = trace.data.spans[0]
        assert span.name == "assistant.run"
        assert span.span_type == SpanType.AGENT
        assert span.inputs == {"task": "1+1"}
        messages = span.outputs["messages"]
        assert len(messages) == 2
        assert (
            messages[0].items()
            >= {
                "content": "1+1",
                "source": "user",
                "models_usage": None,
                "metadata": {},
                "type": "TextMessage",
            }.items()
        )
        assert (
            messages[1].items()
            >= {
                "content": "2",
                "source": "assistant",
                "models_usage": _MODEL_USAGE,
                "metadata": {},
                "type": "TextMessage",
            }.items()
        )

        span = trace.data.spans[1]
        assert span.name == "assistant.on_messages"
        assert span.span_type == SpanType.AGENT
        assert (
            span.outputs["chat_message"].items()
            >= {
                "source": "assistant",
                "models_usage": _MODEL_USAGE,
                "metadata": {},
                "content": "2",
                "type": "TextMessage",
            }.items()
        )

        span = trace.data.spans[2]
        assert span.name == "ReplayChatCompletionClient.create"
        assert span.span_type == SpanType.LLM
        assert span.inputs["messages"] == [
            {"content": _SYSTEM_MESSAGE, "type": "SystemMessage"},
            {"content": "1+1", "source": "user", "type": "UserMessage"},
        ]
        assert span.outputs["content"] == "2"

        assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 6,
            "output_tokens": 1,
            "total_tokens": 7,
        }

        assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "autogen"

        assert traces[0].info.token_usage == {
            "input_tokens": 6,
            "output_tokens": 1,
            "total_tokens": 7,
        }


@pytest.mark.asyncio
async def test_autolog_tool_agent():
    model_client = ReplayChatCompletionClient(
        [
            CreateResult(
                content=[FunctionCall(id="1", arguments='{"number": 1}', name="increment_number")],
                finish_reason="function_calls",
                usage=_MODEL_USAGE,
                cached=False,
            ),
        ],
    )
    model_client.model_info["function_calling"] = True
    TOOL_ATTRIBUTES = [
        {
            "function": {
                "name": "increment_number",
                "description": "Increment a number by 1.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"description": "number", "type": "integer"}},
                    "required": ["number"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
            "type": "function",
        }
    ]

    def increment_number(number: int) -> int:
        """Increment a number by 1."""
        return number + 1

    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message=_SYSTEM_MESSAGE,
        tools=[increment_number],
    )
    mlflow.autogen.autolog()

    await agent.run(task="1+1")

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 3
    span = trace.data.spans[0]
    assert span.name == "assistant.run"
    assert span.span_type == SpanType.AGENT
    assert span.inputs == {"task": "1+1"}
    messages = span.outputs["messages"]
    assert len(messages) == 4
    assert (
        messages[0].items()
        >= {
            "content": "1+1",
            "source": "user",
            "models_usage": None,
            "metadata": {},
            "type": "TextMessage",
        }.items()
    )

    assert (
        messages[1].items()
        >= {
            "content": [
                {
                    "id": "1",
                    "arguments": '{"number": 1}',
                    "name": "increment_number",
                }
            ],
            "source": "assistant",
            "models_usage": _MODEL_USAGE,
            "metadata": {},
            "type": "ToolCallRequestEvent",
        }.items()
    )
    assert (
        messages[2].items()
        >= {
            "content": [
                {
                    "call_id": "1",
                    "content": "2",
                    "is_error": False,
                    "name": "increment_number",
                }
            ],
            "source": "assistant",
            "models_usage": None,
            "metadata": {},
            "type": "ToolCallExecutionEvent",
        }.items()
    )
    assert (
        messages[3].items()
        >= {
            "content": "2",
            "source": "assistant",
            "models_usage": None,
            "metadata": {},
            "type": "ToolCallSummaryMessage",
        }.items()
    )

    span = trace.data.spans[1]
    assert span.name == "assistant.on_messages"
    assert span.span_type == SpanType.AGENT
    assert (
        span.outputs["chat_message"].items()
        >= {
            "source": "assistant",
            "models_usage": None,
            "metadata": {},
            "content": "2",
            "type": "ToolCallSummaryMessage",
        }.items()
    )
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTES

    span = trace.data.spans[2]
    assert span.name == "ReplayChatCompletionClient.create"
    assert span.span_type == SpanType.LLM
    assert span.inputs["messages"] == [
        {"content": _SYSTEM_MESSAGE, "type": "SystemMessage"},
        {"content": "1+1", "source": "user", "type": "UserMessage"},
    ]
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTES
    assert span.outputs["content"] == [
        {"id": "1", "arguments": '{"number": 1}', "name": "increment_number"}
    ]

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 6,
        "output_tokens": 1,
        "total_tokens": 7,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 6,
        "output_tokens": 1,
        "total_tokens": 7,
    }


@pytest.mark.asyncio
async def test_autolog_multi_modal():
    import PIL

    pil_image = PIL.Image.new("RGB", (8, 8))
    img = Image(pil_image)
    user_message = "Can you describe the number in the image?"
    multi_modal_message = MultiModalMessage(content=[user_message, img], source="user")
    model_client = ReplayChatCompletionClient(
        ["2"],
    )
    agent = AssistantAgent("assistant", model_client=model_client, system_message=_SYSTEM_MESSAGE)
    mlflow.autogen.autolog()

    await agent.run(task=multi_modal_message)

    traces = get_traces()

    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 3
    span = trace.data.spans[0]
    assert span.name == "assistant.run"
    assert span.span_type == SpanType.AGENT
    assert span.inputs["task"]["content"][0] == "Can you describe the number in the image?"
    assert "data" in span.inputs["task"]["content"][1]
    messages = span.outputs["messages"]
    assert len(messages) == 2
    assert (
        messages[0].items()
        >= {
            "content": [
                "Can you describe the number in the image?",
                {
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADElEQVR4nGNgGB4AAADIAAGtQHYiAAAAAElFTkSuQmCC",  # noqa: E501
                },
            ],
            "source": "user",
            "models_usage": None,
            "metadata": {},
            "type": "MultiModalMessage",
        }.items()
    )
    assert (
        messages[1].items()
        >= {
            "content": "2",
            "source": "assistant",
            "models_usage": {"completion_tokens": 1, "prompt_tokens": 14},
            "metadata": {},
            "type": "TextMessage",
        }.items()
    )

    span = trace.data.spans[1]
    assert span.name == "assistant.on_messages"
    assert span.span_type == SpanType.AGENT
    assert (
        span.outputs["chat_message"].items()
        >= {
            "source": "assistant",
            "models_usage": {"completion_tokens": 1, "prompt_tokens": 14},
            "metadata": {},
            "content": "2",
            "type": "TextMessage",
        }.items()
    )

    span = trace.data.spans[2]
    assert span.name == "ReplayChatCompletionClient.create"
    assert span.span_type == SpanType.LLM
    assert span.inputs["messages"] == [
        {"content": _SYSTEM_MESSAGE, "type": "SystemMessage"},
        {"content": f"{user_message}\n<image>", "source": "user", "type": "UserMessage"},
    ]
    assert span.outputs["content"] == "2"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 14,
        "output_tokens": 1,
        "total_tokens": 15,
    }

    assert traces[0].info.token_usage == {
        "input_tokens": 14,
        "output_tokens": 1,
        "total_tokens": 15,
    }
