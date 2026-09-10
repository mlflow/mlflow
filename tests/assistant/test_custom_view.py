import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mlflow.assistant.custom_view import (
    CUSTOM_VIEW_RESPONSE_SCHEMA,
    STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA,
    custom_view_response_events,
    parse_custom_view_response,
)
from mlflow.assistant.types import EventType


def test_response_schema_requires_fixed_envelope():
    assert CUSTOM_VIEW_RESPONSE_SCHEMA["required"] == ["type", "text", "title", "messages"]
    assert CUSTOM_VIEW_RESPONSE_SCHEMA["additionalProperties"] is False
    assert STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA["properties"]["messages"]["type"] == "string"


def test_parse_message_response_from_json():
    response = parse_custom_view_response(
        json.dumps({"type": "message", "text": "No change needed.", "title": "", "messages": []})
    )

    assert response.type == "message"
    assert response.text == "No change needed."


def test_parse_message_response_allows_empty_text():
    response = parse_custom_view_response({
        "type": "message",
        "text": "",
        "title": "",
        "messages": [],
    })

    assert response.text == ""


def test_parse_response_with_stringified_messages():
    messages = [{"version": "v0.9", "updateComponents": {}}]
    response = parse_custom_view_response({
        "type": "render_custom_view",
        "text": "Updated the view.",
        "title": "Trace Summary",
        "messages": json.dumps(messages),
    })

    assert response.messages == messages


def test_parse_response_strips_trailing_closing_delimiter_from_complete_messages():
    messages = [{"version": "v0.9", "updateComponents": {}}]
    response = parse_custom_view_response({
        "type": "render_custom_view",
        "text": "Updated the view.",
        "title": "Trace Summary",
        "messages": f"{json.dumps(messages)}}}",
    })

    assert response.messages == messages


@pytest.mark.parametrize(
    "messages",
    ["not-json", "{}", '"text"', "null", f"{json.dumps([{'version': 'v0.9'}])} trailing"],
)
def test_parse_response_rejects_invalid_stringified_messages(messages):
    with pytest.raises(ValidationError, match="messages must be a JSON-encoded array"):
        parse_custom_view_response({
            "type": "render_custom_view",
            "text": "Updated the view.",
            "title": "Trace Summary",
            "messages": messages,
        })


def test_parse_render_response_allows_empty_title():
    response = parse_custom_view_response({
        "type": "render_custom_view",
        "text": "Updated the view.",
        "title": "",
        "messages": [{"version": "v0.9", "updateComponents": {}}],
    })

    assert response.title == ""


@pytest.mark.parametrize("messages", [[], "[]"])
def test_parse_render_response_leaves_empty_messages_for_client_validation(messages):
    response = parse_custom_view_response({
        "type": "render_custom_view",
        "text": "Updated the view.",
        "title": "Trace Summary",
        "messages": messages,
    })

    assert response.messages == []


def test_render_response_emits_terminal_client_tool_call():
    response = parse_custom_view_response({
        "type": "render_custom_view",
        "text": "Updated the view.",
        "title": "Trace Summary",
        "messages": [{"version": "v0.9", "updateComponents": {}}],
    })

    with patch("mlflow.assistant.custom_view.uuid.uuid4", return_value="request-id"):
        events = custom_view_response_events(response)

    assert [event.type for event in events] == [
        EventType.MESSAGE,
        EventType.MESSAGE,
        EventType.CLIENT_TOOL_CALL,
    ]
    assert events[0].data["message"]["content"][0]["text"] == "Updated the view."
    assert events[1].data["message"]["content"][0]["name"] == "render_custom_view"
    assert events[2].data == {
        "request_id": "request-id",
        "tool_name": "render_custom_view",
        "tool_input": {
            "title": "Trace Summary",
            "messages": [{"version": "v0.9", "updateComponents": {}}],
        },
        "continuation": "terminal",
    }
