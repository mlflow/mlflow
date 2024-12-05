import logging
from datetime import datetime
from typing import Any, Literal, NamedTuple, Optional, Union, get_args, get_origin

import pydantic
import pydantic.fields
from packaging.version import Version

from mlflow.exceptions import MlflowException
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
        properties = _infer_fields_from_pydantic_model(type_hint, "Property")
        return ColSpecType(dtype=Object(properties=properties), required=True)
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
    else:
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
        raise MlflowException.invalid_parameter_value(
            f"Unsupported type hint `{type_hint}`, it must include a valid internal type."
        )
    raise MlflowException.invalid_parameter_value(
        f"Unsupported type hint `{type_hint}`, supported types are: "
        f"{list(TYPE_HINTS_TO_DATATYPE_MAPPING.keys())}, pydantic BaseModel subclasses, "
        "lists and dictionaries of primitive types, or typing.Any."
    )


def _infer_fields_from_pydantic_model(
    model: pydantic.BaseModel, field_type: Literal["Property", "ColSpec"]
) -> list[Union[Property, ColSpec]]:
    """
    Infer the fields from a pydantic model.
    If field_type is "Property", the model is seen as an Object, and output fields
    are inferred as Properties.
    If field_type is "ColSpec", the model is seen as a schema, and output fields
    are inferred as ColSpecs.
    """
    if _is_pydantic_type_hint(model):
        fields = model_fields(model)
    else:
        raise TypeError(f"model must be a Pydantic model class, but got {type(model)}")

    if field_type not in ["Property", "ColSpec"]:
        raise MlflowException.invalid_parameter_value(
            f"field_type must be 'Property' or 'ColSpec', but got {field_type}"
        )

    output_fields = []
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
        if field_type == "Property":
            output_fields.append(
                Property(
                    name=field_name,
                    dtype=colspec_type.dtype,
                    required=colspec_type.required,
                )
            )
        elif field_type == "ColSpec":
            output_fields.append(
                ColSpec(
                    type=colspec_type.dtype,
                    name=field_name,
                    required=colspec_type.required,
                )
            )
    if invalid_fields:
        raise MlflowException.invalid_parameter_value(
            "The following fields in the Pydantic model do not have type annotations: "
            f"{invalid_fields}. Please add type annotations to these fields."
        )

    return output_fields


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
    if _is_pydantic_type_hint(type_hint):
        col_specs = _infer_fields_from_pydantic_model(type_hint, "ColSpec")
        return Schema(inputs=col_specs)
    else:
        col_spec_type = _infer_colspec_type_from_type_hint(type_hint)
        # Creating Schema with unnamed optional inputs is not supported
        if col_spec_type.required is False:
            raise MlflowException.invalid_parameter_value(
                "If you would like to use Optional types, use a Pydantic-based type hint "
                "definition."
            )
        return Schema([ColSpec(type=col_spec_type.dtype, required=col_spec_type.required)])


def _validate_example_against_type_hint(example: Any, type_hint: type[Any]) -> None:
    if _is_pydantic_type_hint(type_hint):
        try:
            model_validate(type_hint, example)
        except pydantic.ValidationError as e:
            raise MlflowException.invalid_parameter_value(
                message=f"Input example is not valid for Pydantic model `{type_hint.__name__}`",
            ) from e
    elif type_hint == Any:
        return
    elif type_hint in TYPE_HINTS_TO_DATATYPE_MAPPING:
        if isinstance(example, type_hint):
            return
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
                    return
                if len(args) == 2:
                    effective_type = next((arg for arg in args if arg is not NONE_TYPE), None)
                    return _validate_example_against_type_hint(
                        example=example, type_hint=effective_type
                    )
            # Union type with all valid types is matched as AnyType
            # no validation needed for AnyType
            return
    else:
        _invalid_type_hint_error(type_hint)


def _get_example_validation_error_message(example: Any, type_hint: type[Any]) -> Optional[str]:
    try:
        _validate_example_against_type_hint(example=example, type_hint=type_hint)
    except MlflowException as e:
        return e.message


def _validate_list_elements(element_type: type[Any], example: Any) -> None:
    if not isinstance(example, list):
        raise MlflowException.invalid_parameter_value(
            f"Expected list, but got {type(example).__name__}"
        )
    invalid_elems = {}
    for elem in example:
        if message := _get_example_validation_error_message(example=elem, type_hint=element_type):
            invalid_elems[str(elem)] = message
    if invalid_elems:
        raise MlflowException.invalid_parameter_value(f"Invalid elements in list: {invalid_elems}")


def _validate_dict_elements(element_type: type[Any], example: Any) -> None:
    if not isinstance(example, dict):
        raise MlflowException.invalid_parameter_value(
            f"Expected dict, but got {type(example).__name__}"
        )
    invalid_elems = {}
    for key, value in example.items():
        if not isinstance(key, str):
            invalid_elems[str(key)] = f"Key must be a string, got {type(key).__name__}"
        elif message := _get_example_validation_error_message(
            example=value, type_hint=element_type
        ):
            invalid_elems[key] = message
    if invalid_elems:
        raise MlflowException.invalid_parameter_value(f"Invalid elements in dict: {invalid_elems}")
