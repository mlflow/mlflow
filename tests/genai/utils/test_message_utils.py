import pytest

from mlflow.genai.utils.message_utils import serialize_messages_to_prompts
from mlflow.types.llm import ChatMessage, FunctionToolCallArguments, ToolCall


@pytest.mark.parametrize(
    ("messages", "expected_user_prompt", "expected_system_prompt"),
    [
        # Basic user message (object)
        (
            [ChatMessage(role="user", content="Hello")],
            "Hello",
            None,
        ),
        # Basic user message (dict)
        (
            [{"role": "user", "content": "Hello"}],
            "Hello",
            None,
        ),
        # System + user messages (object)
        (
            [
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
            "Hello",
            "You are helpful.",
        ),
        # System + user messages (dict)
        (
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "Hello",
            "You are helpful.",
        ),
        # Multiple user messages (object)
        (
            [
                ChatMessage(role="user", content="First"),
                ChatMessage(role="user", content="Second"),
            ],
            "First\n\nSecond",
            None,
        ),
        # Multiple user messages (dict)
        (
            [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
            "First\n\nSecond",
            None,
        ),
        # Empty messages
        (
            [],
            "",
            None,
        ),
    ],
    ids=[
        "basic_user_object",
        "basic_user_dict",
        "system_user_object",
        "system_user_dict",
        "multiple_users_object",
        "multiple_users_dict",
        "empty_messages",
    ],
)
def test_serialize_messages_basic(messages, expected_user_prompt, expected_system_prompt):
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == expected_user_prompt
    assert system_prompt == expected_system_prompt


def test_assistant_message_with_content_object():
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\nAssistant: Hi there!"
    assert system_prompt is None


def test_assistant_message_with_content_dict():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\nAssistant: Hi there!"
    assert system_prompt is None


def test_assistant_message_with_tool_calls():
    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="search", arguments='{"query": "test"}')
    )
    messages = [
        ChatMessage(role="user", content="Search for info"),
        ChatMessage(role="assistant", tool_calls=[tool_call]),
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Search for info\n\nAssistant: [Called tools]"
    assert system_prompt is None


def test_assistant_message_with_tool_calls_dict():
    messages = [
        {"role": "user", "content": "Search for info"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {}}]},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Search for info\n\nAssistant: [Called tools]"
    assert system_prompt is None


def test_tool_message_with_name_object():
    messages = [
        ChatMessage(role="user", content="Search"),
        ChatMessage(role="tool", name="search_tool", content='{"results": ["a", "b"]}'),
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == 'Search\n\nTool search_tool: {"results": ["a", "b"]}'
    assert system_prompt is None


def test_tool_message_with_name_dict():
    messages = [
        {"role": "user", "content": "Search"},
        {"role": "tool", "name": "search_tool", "content": '{"results": ["a", "b"]}'},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == 'Search\n\nTool search_tool: {"results": ["a", "b"]}'
    assert system_prompt is None


def test_tool_message_without_name_dict():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "Tool result"},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\ntool: Tool result"
    assert system_prompt is None


def test_custom_role_dict():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "developer", "content": "Custom message"},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Hello\n\ndeveloper: Custom message"
    assert system_prompt is None


def test_full_conversation_object():
    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="search", arguments='{"query": "test"}')
    )
    messages = [
        ChatMessage(role="system", content="Be helpful"),
        ChatMessage(role="user", content="Query"),
        ChatMessage(role="assistant", content="Response"),
        ChatMessage(role="user", content="Search please"),
        ChatMessage(role="assistant", tool_calls=[tool_call]),
        ChatMessage(role="tool", name="search", content="Results"),
        ChatMessage(role="user", content="Follow-up"),
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    expected = (
        "Query\n\nAssistant: Response\n\nSearch please\n\n"
        "Assistant: [Called tools]\n\nTool search: Results\n\nFollow-up"
    )
    assert user_prompt == expected
    assert system_prompt == "Be helpful"


def test_full_conversation_dict():
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Query"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Follow-up"},
    ]
    user_prompt, system_prompt = serialize_messages_to_prompts(messages)
    assert user_prompt == "Query\n\nAssistant: Response\n\nFollow-up"
    assert system_prompt == "Be helpful"
