"""Centralized error classification for MLflow exceptions.

Maps error codes to sqlstate codes and error classes for structured error
classification and observability. Client-side errors use the KAM0x/XXM0x
namespace, while server/CP errors use the KAMCx/XXMCx namespace.

Terminology:
    error_code: The existing MLflow error code from the protobuf definition
        (e.g., INVALID_PARAMETER_VALUE, INTERNAL_ERROR). Defined in
        mlflow/protos/databricks.proto. These are coarse-grained — many
        different failure modes share the same error_code.

    error_class: A more specific classification of the error (e.g.,
        SCHEMA_ENFORCEMENT_FAILED, ATTRIBUTE_NOT_FOUND). Defined in the
        ErrorClass enum below. When an error_class is not explicitly set
        at a raise site, it is auto-derived from the error_code.

    sqlstate: A 5-character code used by reliability dashboards to
        categorize errors (e.g., KAM01, XXMC0). Defined in the SqlState
        enum below. Derived automatically from error_class (if a specific
        mapping exists) or from error_code (generic fallback).

Derivation chain in MlflowException.__init__:
    1. error_class: explicit value if provided, otherwise derived from error_code
    2. sqlstate: explicit value if provided, otherwise derived from error_class
       (via _ERROR_CLASS_TO_SQLSTATE), otherwise derived from error_code
       (via _CLIENT_ERROR_CODE_TO_SQLSTATE)

When to override at a raise site:
    Most raise sites do NOT need to pass sqlstate or error_class — both are
    auto-derived from error_code. Only pass error_class when the error_code
    is too coarse to distinguish the specific failure. For example,
    INVALID_PARAMETER_VALUE is used for both schema enforcement failures and
    attribute lookup failures, so those raise sites pass error_class to
    get distinct sqlstate codes (KAM01 vs KAM04). Never pass sqlstate
    directly — it is always derived from error_class.
"""

from __future__ import annotations

from enum import Enum


class SqlState(str, Enum):
    """SQLSTATE codes for MLflow error classification."""

    # Client system errors (XXM0x)
    CLIENT_INTERNAL_ERROR = "XXM00"

    # Client user errors (KAM0x)
    CLIENT_ATTRIBUTE_NOT_FOUND = "KAM04"
    CLIENT_INVALID_PARAMETER = "KAM00"
    CLIENT_MODEL_SERIALIZATION_FAILED = "KAM03"
    CLIENT_PREDICTION_FUNCTION_FAILED = "KAM02"
    CLIENT_SCHEMA_ENFORCEMENT_FAILED = "KAM01"

    # CP/server system errors (XXMCx)
    CP_INTERNAL_ERROR = "XXMC0"
    CP_INVALID_STATE = "XXMC2"
    CP_TEMPORARILY_UNAVAILABLE = "XXMC1"

    # CP/server user errors (KAMCx)
    CP_INVALID_PARAMETER = "KAMC4"
    CP_PERMISSION_DENIED = "KAMC1"
    CP_REQUEST_RATE_LIMITED = "KAMC3"
    CP_RESOURCE_CONFLICT = "KAMC5"
    CP_RESOURCE_NOT_FOUND = "KAMC2"

    @classmethod
    def from_client_error_code(cls, error_code: str) -> str | None:
        result = _CLIENT_ERROR_CODE_TO_SQLSTATE.get(error_code)
        return result.value if result is not None else None

    @classmethod
    def from_cp_error_code(cls, error_code: str) -> str | None:
        result = _CP_ERROR_CODE_TO_SQLSTATE.get(error_code)
        return result.value if result is not None else None

    @classmethod
    def from_error_class(cls, error_class: str) -> str | None:
        result = _ERROR_CLASS_TO_SQLSTATE.get(error_class)
        return result.value if result is not None else None


class ErrorClass(str, Enum):
    """Error class names for MLflow error classification."""

    # Client error classes
    ATTRIBUTE_NOT_FOUND = "ATTRIBUTE_NOT_FOUND"
    CLIENT_INTERNAL_ERROR = "CLIENT_INTERNAL_ERROR"
    FEATURE_DISABLED = "FEATURE_DISABLED"
    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"
    MODEL_SERIALIZATION_FAILED = "MODEL_SERIALIZATION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PREDICTION_FUNCTION_FAILED = "PREDICTION_FUNCTION_FAILED"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    SCHEMA_ENFORCEMENT_FAILED = "SCHEMA_ENFORCEMENT_FAILED"

    # CP error classes
    CP_INTERNAL_ERROR = "CP_INTERNAL_ERROR"
    CP_INVALID_PARAMETER_VALUE = "CP_INVALID_PARAMETER_VALUE"
    CP_INVALID_STATE = "CP_INVALID_STATE"
    CP_PERMISSION_DENIED = "CP_PERMISSION_DENIED"
    CP_REQUEST_RATE_LIMITED = "CP_REQUEST_RATE_LIMITED"
    CP_RESOURCE_CONFLICT = "CP_RESOURCE_CONFLICT"
    CP_RESOURCE_NOT_FOUND = "CP_RESOURCE_NOT_FOUND"
    CP_TEMPORARILY_UNAVAILABLE = "CP_TEMPORARILY_UNAVAILABLE"

    @classmethod
    def from_client_error_code(cls, error_code: str) -> str | None:
        result = _CLIENT_ERROR_CODE_TO_ERROR_CLASS.get(error_code)
        return result.value if result is not None else None

    @classmethod
    def from_cp_error_code(cls, error_code: str) -> str | None:
        result = _CP_ERROR_CODE_TO_ERROR_CLASS.get(error_code)
        return result.value if result is not None else None


# Client-side mappings: error_code -> sqlstate or error_class
_CLIENT_ERROR_CODE_TO_SQLSTATE: dict[str, SqlState] = {
    "BAD_REQUEST": SqlState.CLIENT_INVALID_PARAMETER,
    "CUSTOMER_UNAUTHORIZED": SqlState.CLIENT_INVALID_PARAMETER,
    "ENDPOINT_NOT_FOUND": SqlState.CLIENT_INVALID_PARAMETER,
    "FEATURE_DISABLED": SqlState.CLIENT_INVALID_PARAMETER,
    "INTERNAL_ERROR": SqlState.CLIENT_INTERNAL_ERROR,
    "INVALID_PARAMETER_VALUE": SqlState.CLIENT_INVALID_PARAMETER,
    "INVALID_STATE": SqlState.CLIENT_INTERNAL_ERROR,
    "NOT_FOUND": SqlState.CLIENT_INVALID_PARAMETER,
    "PERMISSION_DENIED": SqlState.CLIENT_INVALID_PARAMETER,
    "RESOURCE_ALREADY_EXISTS": SqlState.CLIENT_INVALID_PARAMETER,
    "RESOURCE_DOES_NOT_EXIST": SqlState.CLIENT_INVALID_PARAMETER,
    "TEMPORARILY_UNAVAILABLE": SqlState.CLIENT_INTERNAL_ERROR,
}

_CLIENT_ERROR_CODE_TO_ERROR_CLASS: dict[str, ErrorClass] = {
    "BAD_REQUEST": ErrorClass.INVALID_PARAMETER_VALUE,
    "CUSTOMER_UNAUTHORIZED": ErrorClass.PERMISSION_DENIED,
    "ENDPOINT_NOT_FOUND": ErrorClass.RESOURCE_NOT_FOUND,
    "FEATURE_DISABLED": ErrorClass.FEATURE_DISABLED,
    "INTERNAL_ERROR": ErrorClass.CLIENT_INTERNAL_ERROR,
    "INVALID_PARAMETER_VALUE": ErrorClass.INVALID_PARAMETER_VALUE,
    "INVALID_STATE": ErrorClass.CLIENT_INTERNAL_ERROR,
    "NOT_FOUND": ErrorClass.RESOURCE_NOT_FOUND,
    "PERMISSION_DENIED": ErrorClass.PERMISSION_DENIED,
    "RESOURCE_ALREADY_EXISTS": ErrorClass.RESOURCE_ALREADY_EXISTS,
    "RESOURCE_DOES_NOT_EXIST": ErrorClass.RESOURCE_NOT_FOUND,
    "TEMPORARILY_UNAVAILABLE": ErrorClass.CLIENT_INTERNAL_ERROR,
}

# CP/server-side mappings: error_code -> sqlstate or error_class
_CP_ERROR_CODE_TO_SQLSTATE: dict[str, SqlState] = {
    "BAD_REQUEST": SqlState.CP_INVALID_PARAMETER,
    "CUSTOMER_UNAUTHORIZED": SqlState.CP_PERMISSION_DENIED,
    "ENDPOINT_NOT_FOUND": SqlState.CP_RESOURCE_NOT_FOUND,
    "INTERNAL_ERROR": SqlState.CP_INTERNAL_ERROR,
    "INVALID_PARAMETER_VALUE": SqlState.CP_INVALID_PARAMETER,
    "INVALID_STATE": SqlState.CP_INVALID_STATE,
    "NOT_FOUND": SqlState.CP_RESOURCE_NOT_FOUND,
    "PERMISSION_DENIED": SqlState.CP_PERMISSION_DENIED,
    "REQUEST_LIMIT_EXCEEDED": SqlState.CP_REQUEST_RATE_LIMITED,
    "RESOURCE_ALREADY_EXISTS": SqlState.CP_RESOURCE_CONFLICT,
    "RESOURCE_CONFLICT": SqlState.CP_RESOURCE_CONFLICT,
    "RESOURCE_DOES_NOT_EXIST": SqlState.CP_RESOURCE_NOT_FOUND,
    "RESOURCE_EXHAUSTED": SqlState.CP_REQUEST_RATE_LIMITED,
    "TEMPORARILY_UNAVAILABLE": SqlState.CP_TEMPORARILY_UNAVAILABLE,
    "UNAUTHENTICATED": SqlState.CP_PERMISSION_DENIED,
}

_CP_ERROR_CODE_TO_ERROR_CLASS: dict[str, ErrorClass] = {
    "BAD_REQUEST": ErrorClass.CP_INVALID_PARAMETER_VALUE,
    "CUSTOMER_UNAUTHORIZED": ErrorClass.CP_PERMISSION_DENIED,
    "ENDPOINT_NOT_FOUND": ErrorClass.CP_RESOURCE_NOT_FOUND,
    "INTERNAL_ERROR": ErrorClass.CP_INTERNAL_ERROR,
    "INVALID_PARAMETER_VALUE": ErrorClass.CP_INVALID_PARAMETER_VALUE,
    "INVALID_STATE": ErrorClass.CP_INVALID_STATE,
    "NOT_FOUND": ErrorClass.CP_RESOURCE_NOT_FOUND,
    "PERMISSION_DENIED": ErrorClass.CP_PERMISSION_DENIED,
    "REQUEST_LIMIT_EXCEEDED": ErrorClass.CP_REQUEST_RATE_LIMITED,
    "RESOURCE_ALREADY_EXISTS": ErrorClass.CP_RESOURCE_CONFLICT,
    "RESOURCE_CONFLICT": ErrorClass.CP_RESOURCE_CONFLICT,
    "RESOURCE_DOES_NOT_EXIST": ErrorClass.CP_RESOURCE_NOT_FOUND,
    "RESOURCE_EXHAUSTED": ErrorClass.CP_REQUEST_RATE_LIMITED,
    "TEMPORARILY_UNAVAILABLE": ErrorClass.CP_TEMPORARILY_UNAVAILABLE,
    "UNAUTHENTICATED": ErrorClass.CP_PERMISSION_DENIED,
}

# error_class -> sqlstate mapping for specific error patterns that override the
# generic auto-derive. Used at raise sites where the error_code (e.g.,
# INVALID_PARAMETER_VALUE) is too coarse to distinguish the specific failure.
_ERROR_CLASS_TO_SQLSTATE: dict[str, SqlState] = {
    ErrorClass.ATTRIBUTE_NOT_FOUND: SqlState.CLIENT_ATTRIBUTE_NOT_FOUND,
    ErrorClass.MODEL_SERIALIZATION_FAILED: SqlState.CLIENT_MODEL_SERIALIZATION_FAILED,
    ErrorClass.PREDICTION_FUNCTION_FAILED: SqlState.CLIENT_PREDICTION_FUNCTION_FAILED,
    ErrorClass.SCHEMA_ENFORCEMENT_FAILED: SqlState.CLIENT_SCHEMA_ENFORCEMENT_FAILED,
}
