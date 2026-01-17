import json
from unittest import mock

import litellm
import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    _run_databricks_agentic_loop,
    call_chat_completions,
    create_litellm_message_from_databricks_response,
    serialize_messages_to_databricks_prompts,
)
from mlflow.types.llm import ChatMessage
from mlflow.utils import AttrDict


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


@pytest.fixture
def mock_databricks_rag_eval():
    mock_rag_client = mock.MagicMock()
    mock_rag_client.get_chat_completions_result.return_value = AttrDict(
        {"output": "test response", "error_message": None}
    )

    mock_context = mock.MagicMock()
    mock_context.get_context.return_value.build_managed_rag_client.return_value = mock_rag_client
    mock_context.eval_context = lambda func: func

    mock_env_vars = mock.MagicMock()

    mock_module = mock.MagicMock()
    mock_module.context = mock_context
    mock_module.env_vars = mock_env_vars

    return {"module": mock_module, "rag_client": mock_rag_client, "env_vars": mock_env_vars}


@pytest.mark.parametrize(
    ("user_prompt", "system_prompt"),
    [
        ("test user prompt", "test system prompt"),
        ("user prompt only", None),
    ],
)
def test_call_chat_completions_success(user_prompt, system_prompt, mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.VERSION", "1.0.0"
        ),
    ):
        result = call_chat_completions(user_prompt, system_prompt)

        # Verify the client name was set
        mock_databricks_rag_eval[
            "env_vars"
        ].RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set.assert_called_once_with("mlflow-v1.0.0")

        # Verify the managed RAG client was called with correct parameters
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        assert result.output == "test response"


def test_call_chat_completions_with_custom_session_name(mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.VERSION", "1.0.0"
        ),
    ):
        custom_session_name = "custom-session-name"
        result = call_chat_completions(
            "test prompt", "system prompt", session_name=custom_session_name
        )

        # Verify the custom session name was set
        mock_databricks_rag_eval[
            "env_vars"
        ].RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set.assert_called_once_with(custom_session_name)

        # Verify the managed RAG client was called with correct parameters
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt="system prompt",
        )

        assert result.output == "test response"


def test_call_chat_completions_client_error(mock_databricks_rag_eval):
    mock_databricks_rag_eval["rag_client"].get_chat_completions_result.side_effect = RuntimeError(
        "RAG client failed"
    )

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
    ):
        with pytest.raises(RuntimeError, match="RAG client failed"):
            call_chat_completions("test prompt", "system prompt")


def test_call_chat_completions_with_use_case_supported():
    call_args = None

    # Create a mock client with the real method (not a MagicMock) so inspect.signature works
    class MockRAGClient:
        def get_chat_completions_result(self, user_prompt, system_prompt, use_case=None):
            nonlocal call_args
            call_args = {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "use_case": use_case,
            }
            return AttrDict({"output": "test response", "error_message": None})

    mock_context = mock.MagicMock()
    mock_context.get_context.return_value.build_managed_rag_client.return_value = MockRAGClient()
    mock_context.eval_context = lambda func: func

    mock_env_vars = mock.MagicMock()

    mock_module = mock.MagicMock()
    mock_module.context = mock_context
    mock_module.env_vars = mock_env_vars

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_module}),
    ):
        result = call_chat_completions("test prompt", "system prompt", use_case="judge_alignment")

        # Verify use_case was passed since the method signature supports it
        assert call_args == {
            "user_prompt": "test prompt",
            "system_prompt": "system prompt",
            "use_case": "judge_alignment",
        }

        assert result.output == "test response"


def test_call_chat_completions_with_use_case_not_supported(mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
    ):
        # Even though we pass use_case, it won't be forwarded since the mock doesn't support it
        result = call_chat_completions("test prompt", "system prompt", use_case="judge_alignment")

        # Verify use_case was NOT passed (backward compatibility)
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt="system prompt",
        )

        assert result.output == "test response"


# Tests for _run_databricks_agentic_loop


def test_agentic_loop_final_answer_without_tool_calls():
    # Mock response with no tool calls (final answer)
    mock_response = AttrDict(
        {
            "output_json": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"result": "yes", "rationale": "Looks good"}',
                            }
                        }
                    ]
                }
            )
        }
    )

    messages = [litellm.Message(role="user", content="Test prompt")]
    callback_called_with = []

    def callback(content):
        callback_called_with.append(content)
        return {"parsed": content}

    with mock.patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
        return_value=mock_response,
    ) as mock_call:
        result = _run_databricks_agentic_loop(
            messages=messages,
            trace=None,
            on_final_answer=callback,
        )

        # Verify callback was called with the content
        assert len(callback_called_with) == 1
        assert callback_called_with[0] == '{"result": "yes", "rationale": "Looks good"}'
        assert result == {"parsed": '{"result": "yes", "rationale": "Looks good"}'}

        # Verify call_chat_completions was called once (no loop)
        assert mock_call.call_count == 1


def test_agentic_loop_tool_calling_loop(mock_trace):
    # First response has tool calls, second response is final answer
    mock_responses = [
        # First call: LLM requests tool call
        AttrDict(
            {
                "output_json": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_123",
                                            "type": "function",
                                            "function": {
                                                "name": "get_root_span",
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
            }
        ),
        # Second call: LLM returns final answer
        AttrDict(
            {
                "output_json": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"outputs": "The answer is 42"}',
                                }
                            }
                        ]
                    }
                )
            }
        ),
    ]

    messages = [litellm.Message(role="user", content="Extract outputs")]

    def callback(content):
        return {"result": content}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
            side_effect=mock_responses,
        ) as mock_call,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._process_tool_calls"
        ) as mock_process,
    ):
        mock_process.return_value = [
            litellm.Message(
                role="tool",
                content='{"name": "root_span", "inputs": null}',
                tool_call_id="call_123",
                name="get_root_span",
            )
        ]

        result = _run_databricks_agentic_loop(
            messages=messages,
            trace=mock_trace,
            on_final_answer=callback,
        )

        # Verify we looped twice
        assert mock_call.call_count == 2

        # Verify tool calls were processed
        mock_process.assert_called_once()

        # Verify final result
        assert result == {"result": '{"outputs": "The answer is 42"}'}


def test_agentic_loop_max_iteration_limit(mock_trace, monkeypatch):
    # Always return tool calls (never a final answer)
    mock_response = AttrDict(
        {
            "output_json": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_root_span",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        }
    )

    messages = [litellm.Message(role="user", content="Extract outputs")]

    def callback(content):
        return content

    # Set max iterations to 1 for minimal test run
    monkeypatch.setenv("MLFLOW_JUDGE_MAX_ITERATIONS", "1")

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
            return_value=mock_response,
        ) as mock_call,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._process_tool_calls"
        ) as mock_process,
    ):
        mock_process.return_value = [
            litellm.Message(
                role="tool", content="{}", tool_call_id="call_123", name="get_root_span"
            )
        ]

        with pytest.raises(MlflowException, match="iteration limit of 1 exceeded"):
            _run_databricks_agentic_loop(
                messages=messages,
                trace=mock_trace,
                on_final_answer=callback,
            )

        # Verify we hit the limit (called once before raising)
        assert mock_call.call_count == 1


def _create_message(message_type, role, content=None, **kwargs):
    if message_type == "litellm":
        return litellm.Message(role=role, content=content, **kwargs)
    else:
        return ChatMessage(role=role, content=content, **kwargs)


def _create_tool_call(message_type, tool_id, function_name, arguments):
    if message_type == "litellm":
        return litellm.ChatCompletionMessageToolCall(
            id=tool_id,
            type="function",
            function=litellm.Function(name=function_name, arguments=arguments),
        )
    else:
        from mlflow.types.llm import FunctionToolCallArguments, ToolCall

        return ToolCall(
            id=tool_id,
            type="function",
            function=FunctionToolCallArguments(name=function_name, arguments=arguments),
        )


@pytest.mark.parametrize("message_type", ["litellm", "chatmessage"])
@pytest.mark.parametrize(
    ("messages_builder", "expected_user_prompt", "expected_system_prompt"),
    [
        # System message is extracted separately, user message becomes user_prompt
        (
            lambda mt: [
                _create_message(mt, "system", "You are a helpful assistant"),
                _create_message(mt, "user", "What is 2+2?"),
            ],
            "What is 2+2?",
            "You are a helpful assistant",
        ),
        # No system message results in None for system_prompt
        (
            lambda mt: [_create_message(mt, "user", "What is 2+2?")],
            "What is 2+2?",
            None,
        ),
        # Multiple user messages are concatenated with \n\n separator
        (
            lambda mt: [
                _create_message(mt, "user", "First question"),
                _create_message(mt, "user", "Second question"),
            ],
            "First question\n\nSecond question",
            None,
        ),
        # Assistant messages are prefixed with "Assistant: " in the user_prompt
        (
            lambda mt: [
                _create_message(mt, "user", "What is 2+2?"),
                _create_message(mt, "assistant", "4"),
                _create_message(mt, "user", "What is 3+3?"),
            ],
            "What is 2+2?\n\nAssistant: 4\n\nWhat is 3+3?",
            None,
        ),
        # Assistant with tool calls shows "[Called tools]" placeholder
        (
            lambda mt: [
                _create_message(mt, "user", "Get the root span"),
                _create_message(
                    mt,
                    "assistant",
                    None,
                    tool_calls=[_create_tool_call(mt, "call_123", "get_root_span", "{}")],
                ),
            ],
            "Get the root span\n\nAssistant: [Called tools]",
            None,
        ),
        # Tool messages are formatted as "Tool {name}: {content}"
        (
            lambda mt: [
                _create_message(mt, "user", "Get the root span"),
                _create_message(
                    mt,
                    "assistant",
                    None,
                    tool_calls=[_create_tool_call(mt, "call_123", "get_root_span", "{}")],
                ),
                _create_message(
                    mt,
                    "tool",
                    '{"name": "root_span"}',
                    tool_call_id="call_123",
                    name="get_root_span",
                ),
            ],
            "Get the root span\n\nAssistant: [Called tools]\n\nTool get_root_span: "
            '{"name": "root_span"}',
            None,
        ),
        # Full multi-turn conversation with system prompt
        (
            lambda mt: [
                _create_message(mt, "system", "You are helpful"),
                _create_message(mt, "user", "Question 1"),
                _create_message(mt, "assistant", "Answer 1"),
                _create_message(mt, "user", "Question 2"),
            ],
            "Question 1\n\nAssistant: Answer 1\n\nQuestion 2",
            "You are helpful",
        ),
    ],
    ids=[
        "with_system_message",
        "without_system_message",
        "multiple_user_messages",
        "with_assistant_messages",
        "with_tool_calls",
        "with_tool_messages",
        "full_conversation",
    ],
)
def test_serialize_messages_to_databricks_prompts(
    message_type, messages_builder, expected_user_prompt, expected_system_prompt
):
    messages = messages_builder(message_type)
    user_prompt, system_prompt = serialize_messages_to_databricks_prompts(messages)

    assert user_prompt == expected_user_prompt
    assert system_prompt == expected_system_prompt


@pytest.mark.parametrize(
    ("response_data", "expected_role", "expected_content"),
    [
        # Simple string content is extracted as-is
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a response",
                        }
                    }
                ]
            },
            "assistant",
            "This is a response",
        ),
        # List content with text blocks are concatenated with newlines (reasoning models)
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "First part"},
                                {"type": "text", "text": "Second part"},
                                {"type": "other", "data": "ignored"},
                            ],
                        }
                    }
                ]
            },
            "assistant",
            "First part\nSecond part",
        ),
        # List content with no text blocks results in None
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "other", "data": "no text"}],
                        }
                    }
                ]
            },
            "assistant",
            None,
        ),
    ],
    ids=["string_content", "list_content", "empty_list_content"],
)
def test_create_litellm_message_from_databricks_response(
    response_data, expected_role, expected_content
):
    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == expected_role
    assert message.content == expected_content
    assert message.tool_calls is None


def test_create_litellm_message_from_databricks_response_with_single_tool_call():
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_root_span",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ]
    }

    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == "assistant"
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_123"
    assert message.tool_calls[0].type == "function"
    assert message.tool_calls[0].function.name == "get_root_span"
    assert message.tool_calls[0].function.arguments == "{}"


def test_create_litellm_message_from_databricks_response_with_multiple_tool_calls():
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "list_spans", "arguments": "{}"},
                        },
                    ],
                }
            }
        ]
    }

    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == "assistant"
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 2
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].function.name == "get_root_span"
    assert message.tool_calls[1].id == "call_2"
    assert message.tool_calls[1].function.name == "list_spans"


@pytest.mark.parametrize(
    ("response_data", "expected_error"),
    [
        # Response without choices field raises ValueError
        ({}, "missing 'choices' field"),
        # Response with empty choices array raises ValueError
        ({"choices": []}, "missing 'choices' field"),
    ],
    ids=["missing_choices", "empty_choices"],
)
def test_create_litellm_message_from_databricks_response_errors(response_data, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        create_litellm_message_from_databricks_response(response_data)
