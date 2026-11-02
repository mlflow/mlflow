from types import SimpleNamespace

import pytest

from mlflow.gateway.providers.anthropic import AnthropicAdapter


@pytest.mark.parametrize(
    ("tool_choice", "parallel_tool_calls", "expected_tool_choice"),
    [
        ("auto", False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", False, {"type": "any", "disable_parallel_tool_use": True}),
        (
            {"type": "function", "function": {"name": "extract"}},
            False,
            {"type": "tool", "name": "extract", "disable_parallel_tool_use": True},
        ),
        ("auto", True, {"type": "auto"}),
        ("none", False, {"type": "none"}),
        (None, False, None),
    ],
)
def test_chat_parallel_tool_calls(tool_choice, parallel_tool_calls, expected_tool_choice):
    config = SimpleNamespace(model=SimpleNamespace(name="claude-sonnet-4-6"))
    payload = {
        "messages": [{"role": "user", "content": "extract the fields"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract",
                    "description": "extract",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "parallel_tool_calls": parallel_tool_calls,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    result = AnthropicAdapter.chat_to_model(payload, config)

    assert "parallel_tool_calls" not in result
    assert result.get("tool_choice") == expected_tool_choice
