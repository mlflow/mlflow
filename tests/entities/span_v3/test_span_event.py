from mlflow.entities.span_v3 import SpanEvent


def test_span_event():
    event = SpanEvent(
        time_unix_nano=1234567890,
        name="test_event",
        attributes={"key": "value"},
        dropped_attributes_count=0,
    )
    proto = event.to_proto()
    assert SpanEvent.from_proto(proto) == event

    # Complex attributes
    event = SpanEvent(
        time_unix_nano=1234567890,
        name="test_event",
        attributes={
            "str": "value",
            "list[str]": ["a", "b", "c"],
            "dict[str, bool]": {
                "key": True,
                "key2": False,
            },
            "none": None,
        },
        dropped_attributes_count=0,
    )
    proto = event.to_proto()
    assert SpanEvent.from_proto(proto) == event
