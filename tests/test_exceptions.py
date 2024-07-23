import json
import pickle

from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    ENDPOINT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    IO_ERROR,
    RESOURCE_ALREADY_EXISTS,
)


def test_error_code_constructor():
    assert (
        MlflowException("test", error_code=INVALID_PARAMETER_VALUE).error_code
        == "INVALID_PARAMETER_VALUE"
    )


def test_default_error_code():
    assert MlflowException("test").error_code == "INTERNAL_ERROR"


def test_serialize_to_json():
    mlflow_exception = MlflowException("test")
    deserialized = json.loads(mlflow_exception.serialize_as_json())
    assert deserialized["message"] == "test"
    assert deserialized["error_code"] == "INTERNAL_ERROR"


def test_get_http_status_code():
    assert MlflowException("test default").get_http_status_code() == 500
    assert MlflowException("code not in map", error_code=IO_ERROR).get_http_status_code() == 500
    assert MlflowException("test", error_code=INVALID_STATE).get_http_status_code() == 500
    assert MlflowException("test", error_code=ENDPOINT_NOT_FOUND).get_http_status_code() == 404
    assert MlflowException("test", error_code=INVALID_PARAMETER_VALUE).get_http_status_code() == 400
    assert MlflowException("test", error_code=INTERNAL_ERROR).get_http_status_code() == 500
    assert MlflowException("test", error_code=RESOURCE_ALREADY_EXISTS).get_http_status_code() == 400


def test_invalid_parameter_value():
    mlflow_exception = MlflowException.invalid_parameter_value("test")
    assert mlflow_exception.error_code == "INVALID_PARAMETER_VALUE"


def test_rest_exception():
    mlflow_exception = MlflowException("test", error_code=RESOURCE_ALREADY_EXISTS)
    json_exception = mlflow_exception.serialize_as_json()
    deserialized_rest_exception = RestException(json.loads(json_exception))
    assert deserialized_rest_exception.error_code == "RESOURCE_ALREADY_EXISTS"
    assert "test" in deserialized_rest_exception.message


def test_rest_exception_with_unrecognized_error_code():
    # Test that we can create a RestException with a convertible error code.
    exception = RestException({"error_code": "403", "messages": "something important."})
    assert "something important." in str(exception)
    assert exception.error_code == "PERMISSION_DENIED"
    json.loads(exception.serialize_as_json())

    # Test that we can create a RestException with an unrecognized error code.
    exception = RestException({"error_code": "weird error", "messages": "something important."})
    assert "something important." in str(exception)
    json.loads(exception.serialize_as_json())


def test_rest_exception_pickleable():
    e1 = RestException({"error_code": "INTERNAL_ERROR", "message": "abc"})
    e2 = pickle.loads(pickle.dumps(e1))

    assert e1.error_code == e2.error_code
    assert e1.message == e2.message
