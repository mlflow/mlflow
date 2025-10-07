import langchain
import pytest
from packaging.version import Version

from mlflow.types.responses import ResponsesAgentStreamEvent, output_to_responses_items_stream


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.3.0"),
    reason="Langchain messages are not pydantic v2 prior to langchain 0.3.0",
)
def test_output_to_responses_items_stream_langchain():
    """
    Tests langchain message stream to responses items stream conversion.
    Accounts for:
    - AIMessage w/ and w/o tool calls
    - ToolMessage
    - Filtering out HumanMessage from the stream
    - Message
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        AIMessage(
            content="test text0",
            additional_kwargs={},
            response_metadata={},
            name="query_result",
            id="e0eafab0-f008-49d4-ac0d-f17a70096fe1",
        ),
        AIMessage(
            content="Transferring back to supervisor",
            additional_kwargs={},
            response_metadata={"__is_handoff_back": True},
            name="revenue-genie",
            id="5e88662b-29e7-4659-a521-f8175e7642ee",
            tool_calls=[
                {
                    "name": "transfer_back_to_supervisor",
                    "args": {},
                    "id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Successfully transferred back to supervisor",
            name="transfer_back_to_supervisor",
            id="6fd471d8-57d4-46ec-a21a-9bb20dfda4d3",
            tool_call_id="543a6b6b-dc73-463c-9b6e-5d5a941b7669",
        ),
        HumanMessage(
            content="Which companies do I have revenue data for",
            additional_kwargs={},
            response_metadata={},
            id="43d0cf0a-d687-4302-8562-4f2e09603473",
        ),
        AIMessage(
            content="I'll help you",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "toolu_bdrk_01FtmRmzFm89zDtwYu3xdFkh",
                        "function": {
                            "arguments": "{}",
                            "name": "transfer_to_revenue-genie",
                        },
                        "type": "function",
                    }
                ]
            },
            response_metadata={
                "usage": {
                    "prompt_tokens": 551,
                    "completion_tokens": 73,
                    "total_tokens": 624,
                },
                "prompt_tokens": 551,
                "completion_tokens": 73,
                "total_tokens": 624,
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "model_name": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "finish_reason": "tool_calls",
            },
            name="supervisor",
            id="run--e112332b-ed6d-4e10-b17c-adf637fb67eb-0",
            tool_calls=[
                {
                    "name": "transfer_to_revenue-genie",
                    "args": {},
                    "id": "toolu_bdrk_01FtmRmzFm89zDtwYu3xdFkh",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Successfully transferred to revenue-genie",
            name="transfer_to_revenue-genie",
            id="92acdb97-babc-4239-979e-b16880e7f58f",
            tool_call_id="toolu_bdrk_01FtmRmzFm89zDtwYu3xdFkh",
        ),
        AIMessage(
            content="test text1",
            additional_kwargs={},
            response_metadata={},
            name="query_result",
            id="e0eafab0-f008-49d4-ac0d-f17a70096fe1",
        ),
        AIMessage(
            content="Transferring back to supervisor",
            additional_kwargs={},
            response_metadata={"__is_handoff_back": True},
            name="revenue-genie",
            id="5e88662b-29e7-4659-a521-f8175e7642ee",
            tool_calls=[
                {
                    "name": "transfer_back_to_supervisor",
                    "args": {},
                    "id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Successfully transferred back to supervisor",
            name="transfer_back_to_supervisor",
            id="6fd471d8-57d4-46ec-a21a-9bb20dfda4d3",
            tool_call_id="543a6b6b-dc73-463c-9b6e-5d5a941b7669",
        ),
        AIMessage(
            content="test text2",
            additional_kwargs={},
            response_metadata={
                "usage": {
                    "prompt_tokens": 813,
                    "completion_tokens": 108,
                    "total_tokens": 921,
                },
                "prompt_tokens": 813,
                "completion_tokens": 108,
                "total_tokens": 921,
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "model_name": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "finish_reason": "stop",
            },
            name="supervisor",
            id="run--2622edf9-37b6-4e25-9e97-6351d145b198-0",
        ),
    ]
    expected = [
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "e0eafab0-f008-49d4-ac0d-f17a70096fe1",
                "content": [{"text": "test text0", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "5e88662b-29e7-4659-a521-f8175e7642ee",
                "content": [{"text": "Transferring back to supervisor", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call",
                "id": "5e88662b-29e7-4659-a521-f8175e7642ee",
                "call_id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                "name": "transfer_back_to_supervisor",
                "arguments": "{}",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call_output",
                "call_id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                "output": "Successfully transferred back to supervisor",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "run--e112332b-ed6d-4e10-b17c-adf637fb67eb-0",
                "content": [{"text": "I'll help you", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call",
                "id": "run--e112332b-ed6d-4e10-b17c-adf637fb67eb-0",
                "call_id": "toolu_bdrk_01FtmRmzFm89zDtwYu3xdFkh",
                "name": "transfer_to_revenue-genie",
                "arguments": "{}",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call_output",
                "call_id": "toolu_bdrk_01FtmRmzFm89zDtwYu3xdFkh",
                "output": "Successfully transferred to revenue-genie",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "e0eafab0-f008-49d4-ac0d-f17a70096fe1",
                "content": [{"text": "test text1", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "5e88662b-29e7-4659-a521-f8175e7642ee",
                "content": [{"text": "Transferring back to supervisor", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call",
                "id": "5e88662b-29e7-4659-a521-f8175e7642ee",
                "call_id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                "name": "transfer_back_to_supervisor",
                "arguments": "{}",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "type": "function_call_output",
                "call_id": "543a6b6b-dc73-463c-9b6e-5d5a941b7669",
                "output": "Successfully transferred back to supervisor",
            },
        ),
        ResponsesAgentStreamEvent(
            type="response.output_item.done",
            custom_outputs=None,
            item={
                "id": "run--2622edf9-37b6-4e25-9e97-6351d145b198-0",
                "content": [{"text": "test text2", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
        ),
    ]
    result = list(output_to_responses_items_stream(messages))
    assert result == expected
