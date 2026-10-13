import json

import pytest

from mlflow.assistant.types import Event, EventType


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (NotImplementedError(), "NotImplementedError()"),
        (ValueError(), "ValueError()"),
        (RuntimeError("boom"), "boom"),
        (ValueError("bad value"), "bad value"),
    ],
)
def test_from_exception_never_yields_empty_error(exc, expected):
    event = Event.from_exception(exc)
    assert event.type == EventType.ERROR
    assert event.data["error"] == expected


def test_from_error_includes_session_id_only_when_provided():
    assert Event.from_error("boom").data == {"error": "boom"}
    assert Event.from_error("boom", session_id="provider-session").data == {
        "error": "boom",
        "session_id": "provider-session",
    }


def test_from_client_tool_call_carries_request_id_tool_name_and_input():
    event = Event.from_client_tool_call(
        "req-1", "render_custom_view", {"title": "Trace Summary", "messages": []}
    )
    assert event.type == EventType.CLIENT_TOOL_CALL
    assert event.data == {
        "request_id": "req-1",
        "tool_name": "render_custom_view",
        "tool_input": {"title": "Trace Summary", "messages": []},
    }


def test_client_tool_call_event_serializes_to_a_valid_sse_frame():
    event = Event.from_client_tool_call("req-1", "render_custom_view", {"title": "t"})
    sse = event.to_sse_event()
    assert sse.startswith(f"event: {EventType.CLIENT_TOOL_CALL}\n")
    data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line.removeprefix("data:").strip())
    assert payload == {
        "request_id": "req-1",
        "tool_name": "render_custom_view",
        "tool_input": {"title": "t"},
    }


def test_terminal_client_tool_call_carries_continuation():
    event = Event.from_client_tool_call(
        "req-1", "render_custom_view", {"title": "t"}, continuation="terminal"
    )
    assert event.data["continuation"] == "terminal"
