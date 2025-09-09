import json
from unittest.mock import patch

import pytest

from mlflow.tracing.utils.truncation import _get_truncated_preview


@pytest.fixture(autouse=True)
def patch_max_length():
    # Patch max length to 50 to make tests faster
    with patch("mlflow.tracing.utils.truncation._get_max_length", return_value=50):
        yield


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("short string", "short string"),
        ("{'a': 'b'}", "{'a': 'b'}"),
        ("start" + "a" * 50, "start" + "a" * 42 + "..."),
        (None, ""),
    ],
    ids=["short string", "short json", "long string", "none"],
)
def test_truncate_simple_string(input_str, expected):
    assert _get_truncated_preview(input_str, role="user") == expected


def test_truncate_long_non_message_json():
    input_str = json.dumps(
        {
            "a": "b" + "a" * 30,
            "b": "c" + "a" * 30,
        }
    )
    result = _get_truncated_preview(input_str, role="user")
    assert len(result) == 50
    assert result.startswith('{"a": "b')


_TEST_MESSAGE_HISTORY = [
    {"role": "user", "content": "First"},
    {"role": "assistant", "content": "Second"},
    {"role": "user", "content": "Third" + "a" * 50},
    {"role": "assistant", "content": "Fourth"},
]


@pytest.mark.parametrize(
    "input",
    [
        # ChatCompletion API
        {"messages": _TEST_MESSAGE_HISTORY},
        # Responses API
        {"input": _TEST_MESSAGE_HISTORY},
        # Responses Agent
        {"request": {"input": _TEST_MESSAGE_HISTORY}},
    ],
    ids=["chat_completion", "responses", "responses_agent"],
)
def test_truncate_request_messages(input):
    input_str = json.dumps(input)
    assert _get_truncated_preview(input_str, role="assistant") == "Fourth"
    # Long content should be truncated
    assert _get_truncated_preview(input_str, role="user") == "Third" + "a" * 42 + "..."
    # If non-existing role is provided, return the last message
    assert _get_truncated_preview(input_str, role="system") == "Fourth"


def test_truncate_request_choices():
    input_str = json.dumps(
        {
            "choices": [
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "First" + "a" * 50},
                    "finish_reason": "stop",
                },
            ],
            "object": "chat.completions",
        }
    )
    assert _get_truncated_preview(input_str, role="assistant").startswith("First")


def test_truncate_multi_content_messages():
    # If text content exists, use it
    assert (
        _get_truncated_preview(
            json.dumps(
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "a" * 60}]}]}
            ),
            role="user",
        )
        == "a" * 47 + "..."
    )

    # Ignore non text content
    assert (
        _get_truncated_preview(
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "a" * 60},
                                {"type": "image", "image_url": "http://example.com/image.jpg"},
                            ],
                        },
                    ]
                }
            ),
            role="user",
        )
        == "a" * 47 + "..."
    )

    # If non-text content exists, truncate the full json as-is
    assert _get_truncated_preview(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image_url": "http://example.com/image.jpg" + "a" * 50,
                            }
                        ],
                    },
                ]
            }
        ),
        role="user",
    ).startswith('{"messages":')


def test_truncate_responses_api_output():
    input_str = json.dumps(
        {
            "output": [
                {
                    "type": "message",
                    "id": "test",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "a" * 60}],
                }
            ],
        }
    )

    assert _get_truncated_preview(input_str, role="assistant") == "a" * 47 + "..."
