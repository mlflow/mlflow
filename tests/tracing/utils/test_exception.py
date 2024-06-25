import pytest

from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.utils.exception import raise_as_trace_exception


def test_raise_as_trace_exception():
    @raise_as_trace_exception
    def test_fn():
        raise ValueError("error")

    with pytest.raises(MlflowTracingException, match="error"):
        test_fn()

    @raise_as_trace_exception
    def test_fn_no_raise():
        return 0

    assert test_fn_no_raise() == 0
