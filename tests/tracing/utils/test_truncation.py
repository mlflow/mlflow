import json
from unittest.mock import patch

import pytest

from mlflow.tracing.utils.truncation import truncate_request_response_preview


@pytest.fixture(autouse=True)
def patch_max_length():
    # Patch max length to 50 to make tests faster
    with patch("mlflow.tracing.utils.truncation.TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH", 50):
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
    assert truncate_request_response_preview(input_str, role="user") == expected


def test_truncate_long_non_message_json():
    input_str = json.dumps(
        {
            "a": "b" + "a" * 30,
            "b": "c" + "a" * 30,
        }
    )
    result = truncate_request_response_preview(input_str, role="user")
    assert len(result) == 50
    assert result.startswith('{"a": "b')


def test_truncate_request_messages():
    input_str = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third" + "a" * 50},
                {"role": "assistant", "content": "Fourth"},
            ]
        }
    )
    assert truncate_request_response_preview(input_str, role="assistant") == "Fourth"
    # Long content should be truncated
    assert truncate_request_response_preview(input_str, role="user") == "Third" + "a" * 42 + "..."
    # If non-existing role is provided, return the last message
    assert truncate_request_response_preview(input_str, role="system") == "Fourth"


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
    assert truncate_request_response_preview(input_str, role="assistant").startswith("First")


def test_truncate_multi_content_messages():
    # If text content exists, use it
    assert (
        truncate_request_response_preview(
            json.dumps(
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "a" * 60}]}]}
            ),
            role="user",
        )
        == "a" * 47 + "..."
    )

    # Ignore non text content
    assert (
        truncate_request_response_preview(
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
    assert truncate_request_response_preview(
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
