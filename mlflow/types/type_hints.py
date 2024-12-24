import logging
from datetime import datetime
from typing import Any, NamedTuple, Optional, Union, get_args, get_origin

import pydantic
import pydantic.fields
from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.schema import (
    COLSPEC_TYPES,
    AnyType,
    Array,
    ColSpec,
    DataType,
    Map,
    Object,
    Property,
    Schema,
)

PYDANTIC_V1_OR_OLDER = Version(pydantic.VERSION).major <= 1
FIELD_TYPE = pydantic.fields.ModelField if PYDANTIC_V1_OR_OLDER else pydantic.fields.FieldInfo
_logger = logging.getLogger(__name__)
NONE_TYPE = type(None)
UNION_TYPES = (Union,)
try:
    # this import is only available in Python 3.10+
    from types import UnionType

    UNION_TYPES += (UnionType,)
except ImportError:
    pass


# numpy types are not supported
TYPE_HINTS_TO_DATATYPE_MAPPING = {
    int: DataType.long,
    str: DataType.string,
    bool: DataType.boolean,
    float: DataType.double,
    bytes: DataType.binary,
    datetime: DataType.datetime,
}


class ColSpecType(NamedTuple):
    dtype: COLSPEC_TYPES
    required: bool


class InvalidTypeHintException(MlflowException):
    def __init__(self, type_hint, extra_msg=""):
        super().__init__(
            f"Unsupported type hint `{type_hint}`{extra_msg}. Supported types are: "
            f"{list(TYPE_HINTS_TO_DATATYPE_MAPPING.keys())}, pydantic BaseModel subclasses, "
            "lists and dictionaries of primitive types, or typing.Any.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _infer_colspec_type_from_type_hint(type_hint: type[Any]) -> ColSpecType:
    """
    Infer the ColSpec type from a type hint.
    The inferred dtype should be one of the supported data types in COLSPEC_TYPES.
    """
    if type_hint == Any:
        return ColSpecType(dtype=AnyType(), required=True)
    if datatype := TYPE_HINTS_TO_DATATYPE_MAPPING.get(type_hint):
        return ColSpecType(dtype=datatype, required=True)
    elif _is_pydantic_type_hint(type_hint):
        dtype = _infer_type_from_pydantic_model(type_hint)
        return ColSpecType(dtype=dtype, required=True)
    elif origin_type := get_origin(type_hint):
        args = get_args(type_hint)
        if origin_type is list:
            # a valid list[...] type hint must only contain one argument
            if len(args) == 0:
                raise MlflowException.invalid_parameter_value(
                    f"List type hint must contain the internal type, got {type_hint}"
                )
            elif len(args) > 1:
                raise MlflowException.invalid_parameter_value(
                    f"List type hint must contain only one internal type, got {type_hint}"
                )
            else:
                return ColSpecType(
                    dtype=Array(_infer_colspec_type_from_type_hint(type_hint=args[0]).dtype),
                    required=True,
                )
        if origin_type is dict:
            if len(args) == 2:
                if args[0] != str:
                    raise MlflowException.invalid_parameter_value(
                        f"Dictionary key type must be str, got {args[0]} in type hint {type_hint}"
                    )
                return ColSpecType(
                    dtype=Map(_infer_colspec_type_from_type_hint(type_hint=args[1]).dtype),
                    required=True,
                )
            raise MlflowException.invalid_parameter_value(
                f"Dictionary type hint must contain two internal types, got {type_hint}"
            )
        if origin_type in UNION_TYPES:
            if NONE_TYPE in args:
                # This case shouldn't happen, but added for completeness
                if len(args) < 2:
                    raise MlflowException.invalid_parameter_value(
                        f"Union type hint must contain at least one non-None type, got {type_hint}"
                    )
                # Optional type
                elif len(args) == 2:
                    effective_type = next((arg for arg in args if arg is not NONE_TYPE), None)
                    return ColSpecType(
                        dtype=_infer_colspec_type_from_type_hint(effective_type).dtype,
                        required=False,
                    )
                # Optional Union type
                else:
                    _logger.warning(
                        "Union type hint with multiple non-None types is inferred as AnyType, "
                        "and MLflow doesn't validate the data against its internal types."
                    )
                    return ColSpecType(dtype=AnyType(), required=False)
            # Union type with all valid types is matched as AnyType
            else:
                _logger.warning(
                    "Union type hint is inferred as AnyType, and MLflow doesn't validate the data "
                    "against its internal types."
                )
                return ColSpecType(dtype=AnyType(), required=True)
    _invalid_type_hint_error(type_hint)


def _invalid_type_hint_error(type_hint: type[Any]) -> None:
    if (
        type_hint
        in (
            list,
            dict,
            Optional,
        )
        + UNION_TYPES
    ):
        raise InvalidTypeHintException(
            type_hint=type_hint, extra_msg=", it must include a valid internal type"
        )
    raise InvalidTypeHintException(type_hint=type_hint)


def _infer_type_from_pydantic_model(model: pydantic.BaseModel) -> Object:
    """
    Infer the object schema from a pydantic model.
    """
    if _is_pydantic_type_hint(model):
        fields = model_fields(model)
    else:
        raise TypeError(f"model must be a Pydantic model class, but got {type(model)}")

    properties = []
    invalid_fields = []
    for field_name, field_info in fields.items():
        annotation = field_info.annotation
        # this shouldn't happen since pydantic has checks for missing annotations
        # but added here to avoid potential edge cases
        if annotation is None:
            invalid_fields.append(field_name)
            continue
        colspec_type = _infer_colspec_type_from_type_hint(annotation)
        if colspec_type.required is False and field_required(field_info):
            raise MlflowException.invalid_parameter_value(
                f"Optional field `{field_name}` in Pydantic model `{model.__name__}` "
                "doesn't have a default value. Please set default value to None for this field."
            )
        properties.append(
            Property(
                name=field_name,
                dtype=colspec_type.dtype,
                required=colspec_type.required,
            )
        )
    if invalid_fields:
        raise MlflowException.invalid_parameter_value(
            "The following fields in the Pydantic model do not have type annotations: "
            f"{invalid_fields}. Please add type annotations to these fields."
        )

    return Object(properties=properties)


def _is_pydantic_type_hint(type_hint: type[Any]) -> bool:
    try:
        return issubclass(type_hint, pydantic.BaseModel)
    # inspect.isclass(dict[str, int]) is True, but issubclass raises a TypeError
    except TypeError:
        return False


def model_fields(
    model: pydantic.BaseModel,
) -> dict[str, type[FIELD_TYPE]]:
    if PYDANTIC_V1_OR_OLDER:
        return model.__fields__
    return model.model_fields


def model_validate(model: pydantic.BaseModel, values: Any) -> None:
    if PYDANTIC_V1_OR_OLDER:
        model.validate(values)
    else:
        # use strict mode to avoid any data conversion here
        # e.g. "123" will not be converted to 123 if the type is int
        model.model_validate(values, strict=True)


def field_required(field: type[FIELD_TYPE]) -> bool:
    if PYDANTIC_V1_OR_OLDER:
        return field.required
    return field.is_required()


def _infer_schema_from_type_hint(type_hint: type[Any]) -> Schema:
    col_spec_type = _infer_colspec_type_from_type_hint(type_hint)
    # Creating Schema with unnamed optional inputs is not supported
    if col_spec_type.required is False:
        raise MlflowException.invalid_parameter_value(
            "If you would like to use Optional types, use a Pydantic-based type hint definition."
        )
    return Schema([ColSpec(type=col_spec_type.dtype, required=col_spec_type.required)])


def _validate_example_against_type_hint(example: Any, type_hint: type[Any]) -> Any:
    """
    Validate the example against provided type hint.
    The allowed conversions are:
        dictionary example with Pydantic model type hint -> Pydantic model instance

    Args:
        example: The example to validate
        type_hint: The type hint to validate against
    """
    if _is_pydantic_type_hint(type_hint):
        # if example is a pydantic model instance, convert it to a dictionary for validation
        if isinstance(example, pydantic.BaseModel):
            example_dict = example.dict() if PYDANTIC_V1_OR_OLDER else example.model_dump()
        elif isinstance(example, dict):
            example_dict = example
        else:
            raise MlflowException.invalid_parameter_value(
                "Expecting example to be a dictionary or pydantic model instance for "
                f"Pydantic type hint, got {type(example)}"
            )
        try:
            model_validate(type_hint, example_dict)
        except pydantic.ValidationError as e:
            raise MlflowException.invalid_parameter_value(
                message=f"Input example is not valid for Pydantic model `{type_hint.__name__}`",
            ) from e
        else:
            return type_hint(**example_dict) if isinstance(example, dict) else example
    elif type_hint == Any:
        return example
    elif type_hint in TYPE_HINTS_TO_DATATYPE_MAPPING:
        if isinstance(example, type_hint):
            return example
        raise MlflowException.invalid_parameter_value(
            f"Expected type {type_hint}, but got {type(example).__name__}"
        )
    elif origin_type := get_origin(type_hint):
        args = get_args(type_hint)
        if origin_type is list:
            return _validate_list_elements(element_type=args[0], example=example)
        elif origin_type is dict:
            return _validate_dict_elements(element_type=args[1], example=example)
        elif origin_type in UNION_TYPES:
            # Optional type
            if NONE_TYPE in args:
                if example is None:
                    return example
                if len(args) == 2:
                    effective_type = next((arg for arg in args if arg is not NONE_TYPE), None)
                    return _validate_example_against_type_hint(
                        example=example, type_hint=effective_type
                    )
            # Union type with all valid types is matched as AnyType
            # no validation needed for AnyType
            return example
    _invalid_type_hint_error(type_hint)


class ValidationResult(NamedTuple):
    value: Optional[Any] = None
    error_message: Optional[str] = None


def _get_example_validation_result(example: Any, type_hint: type[Any]) -> ValidationResult:
    try:
        value = _validate_example_against_type_hint(example=example, type_hint=type_hint)
        return ValidationResult(value=value)
    except MlflowException as e:
        return ValidationResult(error_message=e.message)


def _validate_list_elements(element_type: type[Any], example: Any) -> list[Any]:
    if not isinstance(example, list):
        raise MlflowException.invalid_parameter_value(
            f"Expected list, but got {type(example).__name__}"
        )
    invalid_elems = {}
    result = []
    for elem in example:
        validation_result = _get_example_validation_result(example=elem, type_hint=element_type)
        if validation_result.error_message:
            invalid_elems[str(elem)] = validation_result.error_message
        else:
            result.append(validation_result.value)
    if invalid_elems:
        raise MlflowException.invalid_parameter_value(f"Invalid elements in list: {invalid_elems}")
    return result


def _validate_dict_elements(element_type: type[Any], example: Any) -> dict[str, Any]:
    if not isinstance(example, dict):
        raise MlflowException.invalid_parameter_value(
            f"Expected dict, but got {type(example).__name__}"
        )
    invalid_elems = {}
    result = {}
    for key, value in example.items():
        if not isinstance(key, str):
            invalid_elems[str(key)] = f"Key must be a string, got {type(key).__name__}"
            continue
        validation_result = _get_example_validation_result(example=value, type_hint=element_type)
        if validation_result.error_message:
            invalid_elems[key] = validation_result.error_message
        else:
            result[key] = validation_result.value
    if invalid_elems:
        raise MlflowException.invalid_parameter_value(f"Invalid elements in dict: {invalid_elems}")
    return result


def _get_origin_type(type_hint: type[Any]) -> Any:
    """
    Get the origin type of a type hint.
    If the type hint is Union type, return the origin type of the effective type.
    If the type hint is Union type with multiple effective types, return Any.
    """
    origin_type = get_origin(type_hint)
    if origin_type in UNION_TYPES:
        args = get_args(type_hint)
        if NONE_TYPE in args and len(args) == 2:
            effective_type = next((arg for arg in args if arg is not NONE_TYPE), None)
            return _get_origin_type(effective_type)
        else:
            # Union types match Any
            return Any
    return origin_type


def _convert_data_to_type_hint(data: Any, type_hint: type[Any]) -> Any:
    """
    Convert data to the expected format based on the type hint.
    This function should only used in limited situations such as mlflow.evaluate.
    Supported conversions:
        - pandas DataFrame with a single column + list[...] type hint -> list
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame) and type_hint != pd.DataFrame:
        origin_type = _get_origin_type(type_hint)
        if type_hint == Any or origin_type == Any:
            return data
        if len(data.columns) != 1:
            # TODO: remove the warning and raise Exception once the bug [ML-48554] is fixed
            _logger.warning(
                "`predict` function with type hints only supports pandas DataFrame "
                f"with a single column. But got {len(data.columns)} columns. "
                "The data will be converted to a list of the first column."
            )
        if origin_type is not list:
            raise MlflowException(
                "Only `list[...]` or `Any` type hint supports pandas DataFrame input "
                f"with a single column. But got {type_hint}."
            )
        return data.iloc[:, 0].tolist()

    return data
