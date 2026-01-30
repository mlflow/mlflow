from unittest.mock import Mock

from mlflow.genai.utils.message_utils import serialize_messages_to_databricks_prompts


class TestSerializeMessagesToDatabricksPrompts:
    def test_basic_user_message(self):
        msg = Mock()
        msg.role = "user"
        msg.content = "Hello"

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts([msg])

        assert user_prompt == "Hello"
        assert system_prompt is None

    def test_system_message(self):
        sys_msg = Mock()
        sys_msg.role = "system"
        sys_msg.content = "You are helpful."

        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts([sys_msg, user_msg])

        assert user_prompt == "Hello"
        assert system_prompt == "You are helpful."

    def test_assistant_message_with_content(self):
        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        assistant_msg = Mock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "Hi there!"
        assistant_msg.tool_calls = None

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts(
            [user_msg, assistant_msg]
        )

        assert user_prompt == "Hello\n\nAssistant: Hi there!"
        assert system_prompt is None

    def test_assistant_message_with_tool_calls(self):
        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Search for info"

        assistant_msg = Mock()
        assistant_msg.role = "assistant"
        assistant_msg.content = None
        assistant_msg.tool_calls = [Mock()]

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts(
            [user_msg, assistant_msg]
        )

        assert user_prompt == "Search for info\n\nAssistant: [Called tools]"
        assert system_prompt is None

    def test_tool_message(self):
        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "Search"

        tool_msg = Mock()
        tool_msg.role = "tool"
        tool_msg.name = "search_tool"
        tool_msg.content = '{"results": ["a", "b"]}'

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts(
            [user_msg, tool_msg]
        )

        assert user_prompt == 'Search\n\nTool search_tool: {"results": ["a", "b"]}'
        assert system_prompt is None

    def test_multiple_user_messages(self):
        msg1 = Mock()
        msg1.role = "user"
        msg1.content = "First"

        msg2 = Mock()
        msg2.role = "user"
        msg2.content = "Second"

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts([msg1, msg2])

        assert user_prompt == "First\n\nSecond"
        assert system_prompt is None

    def test_full_conversation(self):
        sys_msg = Mock()
        sys_msg.role = "system"
        sys_msg.content = "Be helpful"

        user_msg1 = Mock()
        user_msg1.role = "user"
        user_msg1.content = "Query"

        assistant_msg = Mock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "Response"
        assistant_msg.tool_calls = None

        user_msg2 = Mock()
        user_msg2.role = "user"
        user_msg2.content = "Follow-up"

        user_prompt, system_prompt = serialize_messages_to_databricks_prompts(
            [sys_msg, user_msg1, assistant_msg, user_msg2]
        )

        assert user_prompt == "Query\n\nAssistant: Response\n\nFollow-up"
        assert system_prompt == "Be helpful"

    def test_empty_messages(self):
        user_prompt, system_prompt = serialize_messages_to_databricks_prompts([])

        assert user_prompt == ""
        assert system_prompt is None
