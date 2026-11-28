import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage

import mlflow.anthropic
from mlflow.anthropic.autolog import patched_claude_sdk_init


def test_anthropic_autolog_without_claude_sdk():
    sys.modules.pop("claude_agent_sdk", None)

    with (
        patch.dict(
            "sys.modules",
            {
                "anthropic": MagicMock(__version__="0.35.0"),
                "anthropic.resources": MagicMock(Messages=MagicMock, AsyncMessages=MagicMock),
            },
        ),
        patch("mlflow.anthropic.safe_patch"),
    ):
        mlflow.anthropic.autolog()


def _patch_sdk_init(mock_self, response_messages):
    original_init = MagicMock()

    async def fake_receive_response():
        for msg in response_messages:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(original_init, mock_self)
    return original_init


def test_patched_claude_sdk_init_wraps_receive_response():
    mock_self = MagicMock()

    async def fake_receive_response():
        yield "msg1"

    mock_self.receive_response = fake_receive_response
    original_init = MagicMock()
    patched_claude_sdk_init(original_init, mock_self)

    original_init.assert_called_once_with(mock_self, None)
    assert mock_self.receive_response is not fake_receive_response


@pytest.mark.asyncio
async def test_receive_response_builds_trace():
    mock_self = MagicMock()
    messages = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
        ResultMessage(
            subtype="success",
            duration_ms=5000,
            duration_api_ms=4000,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
    ]
    _patch_sdk_init(mock_self, messages)

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    called_messages = mock_process.call_args[0][0]
    assert len(called_messages) == 3
    result_messages = [m for m in called_messages if isinstance(m, ResultMessage)]
    assert len(result_messages) == 1
    assert result_messages[0].usage == {"input_tokens": 100, "output_tokens": 20}


@pytest.mark.asyncio
async def test_query_captures_async_generator_prompt():
    mock_self = MagicMock()

    async def fake_query(prompt, *args, **kwargs):
        # Consume the generator like the real SDK would
        async for _ in prompt:
            pass

    mock_self.query = fake_query

    response_messages = [
        AssistantMessage(content=[TextBlock(text="Hi!")], model="claude-sonnet-4-20250514"),
        ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="s",
        ),
    ]
    _patch_sdk_init(mock_self, response_messages)

    async def prompt_generator():
        yield {"type": "user", "message": {"role": "user", "content": "Hello from generator"}}

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        await mock_self.query(prompt_generator())
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    called_messages = mock_process.call_args[0][0]
    user_messages = [m for m in called_messages if isinstance(m, UserMessage)]
    assert len(user_messages) == 1
    assert user_messages[0].content == "Hello from generator"


@pytest.mark.asyncio
async def test_receive_response_skips_when_autologging_disabled():
    mock_self = MagicMock()
    _patch_sdk_init(mock_self, ["msg1", "msg2"])

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=True),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_not_called()


def _result_message(input_tokens, output_tokens):
    return ResultMessage(
        subtype="success",
        duration_ms=1000,
        duration_api_ms=800,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


@pytest.mark.asyncio
async def test_receive_response_traces_each_turn_separately():
    mock_self = MagicMock()

    async def fake_query(prompt, *args, **kwargs):
        pass

    mock_self.query = fake_query

    turns = {
        "What is 2 + 2?": [
            AssistantMessage(content=[TextBlock(text="4")], model="claude-sonnet-4-20250514"),
            _result_message(100, 5),
        ],
        "And 3 + 3?": [
            AssistantMessage(content=[TextBlock(text="6")], model="claude-sonnet-4-20250514"),
            _result_message(150, 7),
        ],
    }
    pending = []

    async def fake_receive_response():
        for msg in turns[pending.pop(0)]:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(MagicMock(), mock_self)

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        for prompt in turns:
            pending.append(prompt)
            await mock_self.query(prompt)
            [msg async for msg in mock_self.receive_response()]

    assert mock_process.call_count == 2
    first_turn, second_turn = (c.args[0] for c in mock_process.call_args_list)
    assert len(first_turn) == 3
    assert first_turn[0].content == "What is 2 + 2?"
    assert first_turn[2].usage == {"input_tokens": 100, "output_tokens": 5}
    assert len(second_turn) == 3
    assert second_turn[0].content == "And 3 + 3?"
    assert second_turn[1].content[0].text == "6"
    assert second_turn[2].usage == {"input_tokens": 150, "output_tokens": 7}


@pytest.mark.asyncio
async def test_receive_response_closed_late_keeps_next_turn_prompt():
    mock_self = MagicMock()

    async def fake_query(prompt, *args, **kwargs):
        pass

    mock_self.query = fake_query

    turns = {
        "What is 2 + 2?": [
            AssistantMessage(content=[TextBlock(text="4")], model="claude-sonnet-4-20250514"),
            _result_message(100, 5),
        ],
        "And 3 + 3?": [
            AssistantMessage(content=[TextBlock(text="6")], model="claude-sonnet-4-20250514"),
            _result_message(150, 7),
        ],
    }
    pending = []

    async def fake_receive_response():
        for msg in turns[pending.pop(0)]:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(MagicMock(), mock_self)

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        pending.append("What is 2 + 2?")
        await mock_self.query("What is 2 + 2?")
        first_stream = mock_self.receive_response()
        async for msg in first_stream:
            if isinstance(msg, ResultMessage):
                break

        # Send the next query before the first stream is closed
        pending.append("And 3 + 3?")
        await mock_self.query("And 3 + 3?")
        await first_stream.aclose()
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    second_turn = mock_process.call_args.args[0]
    assert len(second_turn) == 3
    assert second_turn[0].content == "And 3 + 3?"
    assert second_turn[2].usage == {"input_tokens": 150, "output_tokens": 7}


@pytest.mark.asyncio
async def test_receive_response_traces_each_turn_from_one_async_generator_prompt():
    mock_self = MagicMock()
    sent_prompts = asyncio.Queue()

    async def fake_query(prompt, *args, **kwargs):
        async for item in prompt:
            await sent_prompts.put(item["message"]["content"])

    mock_self.query = fake_query

    turns = {
        "What is 2 + 2?": [
            AssistantMessage(content=[TextBlock(text="4")], model="claude-sonnet-4-20250514"),
            _result_message(100, 5),
        ],
        "And 3 + 3?": [
            AssistantMessage(content=[TextBlock(text="6")], model="claude-sonnet-4-20250514"),
            _result_message(150, 7),
        ],
    }

    async def fake_receive_response():
        for msg in turns[await sent_prompts.get()]:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(MagicMock(), mock_self)

    first_turn_done = asyncio.Event()

    async def prompt_generator():
        yield {"type": "user", "message": {"role": "user", "content": "What is 2 + 2?"}}
        await first_turn_done.wait()
        yield {"type": "user", "message": {"role": "user", "content": "And 3 + 3?"}}

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        query_task = asyncio.create_task(mock_self.query(prompt_generator()))
        [msg async for msg in mock_self.receive_response()]
        first_turn_done.set()
        [msg async for msg in mock_self.receive_response()]
        await query_task

    assert mock_process.call_count == 2
    first_turn, second_turn = (c.args[0] for c in mock_process.call_args_list)
    assert len(first_turn) == 3
    assert first_turn[0].content == "What is 2 + 2?"
    assert first_turn[2].usage == {"input_tokens": 100, "output_tokens": 5}
    assert len(second_turn) == 3
    assert second_turn[0].content == "And 3 + 3?"
    assert second_turn[2].usage == {"input_tokens": 150, "output_tokens": 7}


@pytest.mark.asyncio
async def test_receive_response_failed_before_first_message_drops_its_prompt():
    mock_self = MagicMock()

    async def fake_query(prompt, *args, **kwargs):
        pass

    mock_self.query = fake_query

    responses = [
        None,
        [
            AssistantMessage(content=[TextBlock(text="6")], model="claude-sonnet-4-20250514"),
            _result_message(150, 7),
        ],
    ]

    async def fake_receive_response():
        turn = responses.pop(0)
        if turn is None:
            raise ConnectionError("stream failed")
        for msg in turn:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(MagicMock(), mock_self)

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        await mock_self.query("What is 2 + 2?")
        with pytest.raises(ConnectionError, match="stream failed"):
            [msg async for msg in mock_self.receive_response()]
        await mock_self.query("And 3 + 3?")
        [msg async for msg in mock_self.receive_response()]

    mock_process.assert_called_once()
    second_turn = mock_process.call_args.args[0]
    assert len(second_turn) == 3
    assert second_turn[0].content == "And 3 + 3?"
    assert second_turn[2].usage == {"input_tokens": 150, "output_tokens": 7}


@pytest.mark.asyncio
async def test_receive_response_assigns_queued_async_generator_prompts_to_their_turns():
    mock_self = MagicMock()
    sent_prompts = []

    async def fake_query(prompt, *args, **kwargs):
        sent_prompts.extend([item["message"]["content"] async for item in prompt])

    mock_self.query = fake_query

    turns = {
        "What is 2 + 2?": [
            AssistantMessage(content=[TextBlock(text="4")], model="claude-sonnet-4-20250514"),
            _result_message(100, 5),
        ],
        "And 3 + 3?": [
            AssistantMessage(content=[TextBlock(text="6")], model="claude-sonnet-4-20250514"),
            _result_message(150, 7),
        ],
    }

    async def fake_receive_response():
        for msg in turns[sent_prompts.pop(0)]:
            yield msg

    mock_self.receive_response = fake_receive_response
    patched_claude_sdk_init(MagicMock(), mock_self)

    async def prompt_generator():
        for prompt in turns:
            yield {"type": "user", "message": {"role": "user", "content": prompt}}

    with (
        patch("mlflow.utils.autologging_utils.autologging_is_disabled", return_value=False),
        patch("mlflow.claude_code.tracing.process_sdk_messages") as mock_process,
    ):
        # Both prompts are sent before the first response is consumed
        await mock_self.query(prompt_generator())
        [msg async for msg in mock_self.receive_response()]
        [msg async for msg in mock_self.receive_response()]

    assert mock_process.call_count == 2
    first_turn, second_turn = (c.args[0] for c in mock_process.call_args_list)
    assert len(first_turn) == 3
    assert first_turn[0].content == "What is 2 + 2?"
    assert first_turn[2].usage == {"input_tokens": 100, "output_tokens": 5}
    assert len(second_turn) == 3
    assert second_turn[0].content == "And 3 + 3?"
    assert second_turn[2].usage == {"input_tokens": 150, "output_tokens": 7}
