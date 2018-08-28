from mlflow.exceptions import ExecutionException


def test_execution_exception_string_repr():
    exc = ExecutionException("Uh oh")
    assert str(exc) == "Uh oh"
