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
