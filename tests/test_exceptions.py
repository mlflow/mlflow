import json
import pickle

import pytest

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


def test_rest_exception_with_null_error_code():
    exception = RestException({"error_code": None, "message": "test message"})
    assert exception.error_code == "INTERNAL_ERROR"
    assert "test message" in str(exception)


def test_rest_exception_with_missing_error_code():
    exception = RestException({"message": "test message"})
    assert exception.error_code == "INTERNAL_ERROR"
    assert "test message" in str(exception)


# --- sqlstate / error_class auto-derive tests ---


def test_sqlstate_auto_derived_from_error_code():
    exc = MlflowException("test")
    assert exc.sqlstate == "XXM00"
    assert exc.error_class == "CLIENT_INTERNAL_ERROR"

    exc = MlflowException("test", error_code=INVALID_PARAMETER_VALUE)
    assert exc.sqlstate == "KAM00"
    assert exc.error_class == "INVALID_PARAMETER_VALUE"

    exc = MlflowException("test", error_code=INTERNAL_ERROR)
    assert exc.sqlstate == "XXM00"
    assert exc.error_class == "CLIENT_INTERNAL_ERROR"


def test_sqlstate_explicit_overrides_auto_derive():
    exc = MlflowException(
        "test",
        error_code=INVALID_PARAMETER_VALUE,
        sqlstate="KAM01",
        error_class="SCHEMA_ENFORCEMENT_FAILED",
    )
    assert exc.sqlstate == "KAM01"
    assert exc.error_class == "SCHEMA_ENFORCEMENT_FAILED"


def test_sqlstate_serialize_as_json_includes_auto_derived():
    exc = MlflowException("test", error_code=INVALID_PARAMETER_VALUE)
    deserialized = json.loads(exc.serialize_as_json())
    assert deserialized["sqlstate"] == "KAM00"
    assert deserialized["error_class"] == "INVALID_PARAMETER_VALUE"


def test_sqlstate_serialize_as_json_includes_explicit():
    exc = MlflowException("test", sqlstate="KAM01", error_class="SCHEMA_ENFORCEMENT_FAILED")
    deserialized = json.loads(exc.serialize_as_json())
    assert deserialized["sqlstate"] == "KAM01"
    assert deserialized["error_class"] == "SCHEMA_ENFORCEMENT_FAILED"


def test_sqlstate_none_for_unknown_error_code():
    exc = MlflowException("test", error_code=IO_ERROR)
    assert exc.sqlstate is None
    assert exc.error_class is None
    deserialized = json.loads(exc.serialize_as_json())
    assert "sqlstate" not in deserialized
    assert "error_class" not in deserialized


def test_invalid_parameter_value_auto_derives_sqlstate():
    exc = MlflowException.invalid_parameter_value("bad input")
    assert exc.error_code == "INVALID_PARAMETER_VALUE"
    assert exc.sqlstate == "KAM00"
    assert exc.error_class == "INVALID_PARAMETER_VALUE"


def test_invalid_parameter_value_with_explicit_override():
    exc = MlflowException.invalid_parameter_value(
        "bad input", sqlstate="KAM02", error_class="PREDICTION_FUNCTION_FAILED"
    )
    assert exc.error_code == "INVALID_PARAMETER_VALUE"
    assert exc.sqlstate == "KAM02"
    assert exc.error_class == "PREDICTION_FUNCTION_FAILED"


# --- RestException CP mapping tests ---


@pytest.mark.parametrize(
    ("error_code", "expected_sqlstate", "expected_error_class"),
    [
        ("PERMISSION_DENIED", "KAMC1", "CP_PERMISSION_DENIED"),
        ("RESOURCE_DOES_NOT_EXIST", "KAMC2", "CP_RESOURCE_NOT_FOUND"),
        ("REQUEST_LIMIT_EXCEEDED", "KAMC3", "CP_REQUEST_RATE_LIMITED"),
        ("INVALID_PARAMETER_VALUE", "KAMC4", "CP_INVALID_PARAMETER_VALUE"),
        ("INTERNAL_ERROR", "XXMC0", "CP_INTERNAL_ERROR"),
        ("TEMPORARILY_UNAVAILABLE", "XXMC1", "CP_TEMPORARILY_UNAVAILABLE"),
        ("INVALID_STATE", "XXMC2", "CP_INVALID_STATE"),
    ],
)
def test_rest_exception_cp_sqlstate_mapping(error_code, expected_sqlstate, expected_error_class):
    exc = RestException({"error_code": error_code, "message": "test"})
    assert exc.sqlstate == expected_sqlstate
    assert exc.error_class == expected_error_class


def test_rest_exception_preserves_sqlstate_from_json():
    exc = RestException({
        "error_code": "PERMISSION_DENIED",
        "message": "no access",
        "sqlstate": "CUSTOM",
        "error_class": "CUSTOM_CLASS",
    })
    assert exc.sqlstate == "CUSTOM"
    assert exc.error_class == "CUSTOM_CLASS"


def test_rest_exception_ignores_null_sqlstate_from_json():
    exc = RestException({
        "error_code": "PERMISSION_DENIED",
        "message": "no access",
        "sqlstate": None,
    })
    assert exc.sqlstate == "KAMC1"


def test_rest_exception_pickle_with_sqlstate():
    e1 = RestException({"error_code": "PERMISSION_DENIED", "message": "no access"})
    e2 = pickle.loads(pickle.dumps(e1))
    assert e1.error_code == e2.error_code
    assert e1.message == e2.message
    assert e1.sqlstate == e2.sqlstate
    assert e1.error_class == e2.error_class


def test_rest_exception_unrecognized_error_code():
    exc = RestException({"error_code": "weird error", "messages": "something"})
    # Unrecognized error codes fall back to INTERNAL_ERROR, which maps to XXMC0
    assert exc.sqlstate == "XXMC0"
    assert exc.error_class == "CP_INTERNAL_ERROR"
