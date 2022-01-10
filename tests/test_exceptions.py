import json

from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    ENDPOINT_NOT_FOUND,
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
    IO_ERROR,
)


class TestMlflowException:
    def test_error_code_constructor(self):
        assert (
            MlflowException("test", error_code=INVALID_PARAMETER_VALUE).error_code
            == "INVALID_PARAMETER_VALUE"
        )

    def test_default_error_code(self):
        assert MlflowException("test").error_code == "INTERNAL_ERROR"

    def test_serialize_to_json(self):
        mlflow_exception = MlflowException("test")
        deserialized = json.loads(mlflow_exception.serialize_as_json())
        assert deserialized["message"] == "test"
        assert deserialized["error_code"] == "INTERNAL_ERROR"

    def test_get_http_status_code(self):
        assert MlflowException("test default").get_http_status_code() == 500
        assert MlflowException("code not in map", error_code=IO_ERROR).get_http_status_code() == 500
        assert MlflowException("test", error_code=INVALID_STATE).get_http_status_code() == 500
        assert MlflowException("test", error_code=ENDPOINT_NOT_FOUND).get_http_status_code() == 404
        assert (
            MlflowException("test", error_code=INVALID_PARAMETER_VALUE).get_http_status_code()
            == 400
        )
        assert MlflowException("test", error_code=INTERNAL_ERROR).get_http_status_code() == 500
        assert (
            MlflowException("test", error_code=RESOURCE_ALREADY_EXISTS).get_http_status_code()
            == 400
        )


def test_rest_exception():
    mlflow_exception = MlflowException("test", error_code=RESOURCE_ALREADY_EXISTS)
    json_exception = mlflow_exception.serialize_as_json()
    deserialized_rest_exception = RestException(json.loads(json_exception))
    assert deserialized_rest_exception.error_code == "RESOURCE_ALREADY_EXISTS"
    assert "test" in deserialized_rest_exception.message
