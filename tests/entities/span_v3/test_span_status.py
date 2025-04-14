from mlflow.entities.span_v3.span_status import SpanStatus
from mlflow.entities.span_v3.span_status_code import SpanStatusCode


def test_span_status():
    status = SpanStatus(
        message="test",
        code=SpanStatusCode.STATUS_CODE_OK,
    )
    proto = status.to_proto()
    assert SpanStatus.from_proto(proto) == status
