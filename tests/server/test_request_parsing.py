from unittest import mock

from mlflow.protos.service_pb2 import TestBooleanParams as BooleanParamsProto
from mlflow.server.handlers import _get_request_message


def test_can_parse_get_boolean_true():
    request = mock.MagicMock()
    request.method = "GET"
    request.args = {"flag_true": "true", "text_field": "hello"}
    msg = _get_request_message(BooleanParamsProto(), flask_request=request)
    assert msg.flag_true is True
    assert msg.text_field == "hello"


def test_can_parse_get_boolean_false():
    request = mock.MagicMock()
    request.method = "GET"
    request.args = {"flag_false": "false", "text_field": "world"}
    msg = _get_request_message(BooleanParamsProto(), flask_request=request)
    assert msg.flag_false is False
    assert msg.text_field == "world"


def test_can_parse_get_boolean_case_insensitive():
    request = mock.MagicMock()
    request.method = "GET"
    request.args = {"flag_true": "True", "flag_false": "FALSE"}
    msg = _get_request_message(BooleanParamsProto(), flask_request=request)
    assert msg.flag_true is True
    assert msg.flag_false is False
