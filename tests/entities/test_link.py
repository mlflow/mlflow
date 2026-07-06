from mlflow.entities.link import Link

TRACE_ID = "tr-abc123"
SPAN_ID = "aabbccddeeff0011"


def test_link_to_dict():
    link = Link(trace_id=TRACE_ID, span_id=SPAN_ID, attributes={"type": "causality"})
    assert link.to_dict() == {
        "trace_id": TRACE_ID,
        "span_id": SPAN_ID,
        "attributes": {"type": "causality"},
    }


def test_link_from_dict():
    data = {"trace_id": TRACE_ID, "span_id": SPAN_ID, "attributes": {"type": "test"}}
    link = Link.from_dict(data)
    assert link.trace_id == TRACE_ID
    assert link.span_id == SPAN_ID
    assert link.attributes == {"type": "test"}


def test_link_from_dict_without_attributes():
    data = {"trace_id": TRACE_ID, "span_id": SPAN_ID}
    link = Link.from_dict(data)
    assert link.trace_id == TRACE_ID
    assert link.span_id == SPAN_ID
    assert link.attributes is None
