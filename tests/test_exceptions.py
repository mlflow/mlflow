import json

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, TEMPORARILY_UNAVAILABLE, \
    ENDPOINT_NOT_FOUND, INTERNAL_ERROR


class TestMlflowException(object):
    def test_error_code_constructor(self):
        assert MlflowException('test', error_code=INVALID_PARAMETER_VALUE).error_code == \
               'INVALID_PARAMETER_VALUE'

    def test_default_error_code(self):
        assert MlflowException('test').error_code == 'INTERNAL_ERROR'

    def test_serialize_to_json(self):
        mlflow_exception = MlflowException('test')
        deserialized = json.loads(mlflow_exception.serialize_as_json())
        assert deserialized['message'] == 'test'
        assert deserialized['error_code'] == 'INTERNAL_ERROR'

    def test_get_http_status_code(self):
        assert MlflowException('text', error_code=TEMPORARILY_UNAVAILABLE).get_http_status_code() \
             == 503
        assert MlflowException('text', error_code=ENDPOINT_NOT_FOUND).get_http_status_code() \
            == 404
        assert MlflowException('test', error_code=INVALID_PARAMETER_VALUE).get_http_status_code() \
            == 400
        assert MlflowException('test', error_code=INTERNAL_ERROR).get_http_status_code() \
            == 500
