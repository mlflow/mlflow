import json
from mlflow.exceptions import ExecutionException, RestException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode


def test_execution_exception_string_repr():
    exc = ExecutionException("Uh oh")
    assert str(exc) == "Uh oh"
    json.loads(exc.serialize_as_json())


def test_rest_exception_default_error_code():
    exc = RestException({"message": "something important."})
    assert "something important." in str(exc)


def test_rest_exception_error_code_is_not_none():
    error_string = "something important."
    exc = RestException({"message": error_string})
    assert "None" not in error_string
    assert "None" not in str(exc)
    json.loads(exc.serialize_as_json())


def test_rest_exception_without_message():
    exc = RestException({"my_property": "something important."})
    assert "something important." in str(exc)
    json.loads(exc.serialize_as_json())


def test_rest_exception_error_code_and_no_message():
    exc = RestException(
        {"error_code": ErrorCode.Name(RESOURCE_DOES_NOT_EXIST), "messages": "something important."}
    )
    assert "something important." in str(exc)
    assert "RESOURCE_DOES_NOT_EXIST" in str(exc)
    json.loads(exc.serialize_as_json())
