import base64
import logging
from datetime import datetime
from functools import lru_cache
from types import UnionType
from typing import Any, NamedTuple, Optional, TypeVar, Union, get_args, get_origin

import pydantic
import pydantic.fields

from mlflow.environment_variables import _MLFLOW_IS_IN_SERVING_ENVIRONMENT
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
from mlflow.utils.warnings_utils import color_warning

FIELD_TYPE = pydantic.fields.FieldInfo
NONE_TYPE = type(None)
UNION_TYPES = (Union, UnionType)

_logger = logging.getLogger(__name__)
# special type hint that can be used to convert data to
# the input example type after data validation
TypeFromExample = TypeVar("TypeFromExample")
OPTIONAL_INPUT_MSG = (
    "Input cannot be Optional type. Fix this by removing the "
    "Optional wrapper from the type hint. To use optional fields, "
    "use a Pydantic-based type hint definition. See "
    "https://docs.pydantic.dev/latest/api/base_model/ for pydantic "
    "BaseModel examples. Check https://mlflow.org/docs/latest/model/python_model.html#supported-type-hints"
    " for more details."
)

# numpy types are not supported
TYPE_HINTS_TO_DATATYPE_MAPPING = {
    int: DataType.long,
    str: DataType.string,
    bool: DataType.boolean,
    float: DataType.double,
    bytes: DataType.binary,
    datetime: DataType.datetime,
}

SUPPORTED_TYPE_HINT_MSG = (
    "Type hints must be a list[...] where collection element type is one of these types: "
    f"{list(TYPE_HINTS_TO_DATATYPE_MAPPING.keys())}, pydantic BaseModel subclasses, "
    "lists and dictionaries of primitive types, or typing.Any. Check "
    "https://mlflow.org/docs/latest/model/python_model.html#supported-type-hints for more details."
)


def _try_import_numpy():
    try:
        import numpy

        return numpy
    except ImportError:
        return


@lru_cache(maxsize=1)
def type_hints_no_signature_inference():
    """
    This function returns a tuple of types that can be used
    as type hints, but no schema can be inferred from them.

    ..note::
        These types can not be used as nested types in other type hints.
    """
    type_hints = ()
    try:
        import pandas as pd

        type_hints += (
            pd.DataFrame,
            pd.Series,
        )
    except ImportError:
        pass

    try:
        import numpy as np

        type_hints += (np.ndarray,)
    except ImportError:
        pass

    try:
        from scipy.sparse import csc_matrix, csr_matrix

        type_hints += (csc_matrix, csr_matrix)
    except ImportError:
        pass

    return type_hints


class ColSpecType(NamedTuple):
    dtype: COLSPEC_TYPES
    required: bool


class UnsupportedTypeHintException(MlflowException):
    def __init__(self, type_hint):
        super().__init__(
            f"Unsupported type hint `{_type_hint_repr(type_hint)}`. {SUPPORTED_TYPE_HINT_MSG}",
            error_code=INVALID_PARAMETER_VALUE,
        )


class InvalidTypeHintException(MlflowException):
    def __init__(self, *, message):
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE)


def _signature_cannot_be_inferred_from_type_hint(type_hint: type[Any]) -> bool:
    return type_hint in type_hints_no_signature_inference()


def _is_type_hint_from_example(type_hint: type[Any]) -> bool:
    return type_hint == TypeFromExample


def _is_example_valid_for_type_from_example(example: Any) -> bool:
    allowed_types = (list,)
    try:
        import pandas as pd

        allowed_types += (pd.DataFrame, pd.Series)
    except ImportError:
        pass
    return isinstance(example, allowed_types)


def _convert_dataframe_to_example_format(data: Any, input_example: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if isinstance(input_example, pd.DataFrame):
            return data
        if isinstance(input_example, pd.Series):
            data = data.iloc[:, 0]
            data.name = input_example.name
            return data
        if np.isscalar(input_example):
            return data.iloc[0, 0]
        if isinstance(input_example, dict):
            if len(data) == 1:
                return data.to_dict(orient="records")[0]
            else:
                # This case shouldn't happen
                _logger.warning("Cannot convert DataFrame to a single dictionary.")
                return data
        if isinstance(input_example, list):
            # list[scalar]
            if len(data.columns) == 1 and all(np.isscalar(x) for x in input_example):
                return data.iloc[:, 0].tolist()
            else:
                # NB: there are some cases that this doesn't work well, but it's the best we can do
                # e.g. list of dictionaries with different keys
                # [{"a": 1}, {"b": 2}] -> pd.DataFrame(...) during schema enforcement
                # here -> [{'a': 1.0, 'b': nan}, {'a': nan, 'b': 2.0}]
                return data.to_dict(orient="records")

    return data


def _infer_colspec_type_from_type_hint(type_hint: type[Any]) -> ColSpecType:
    """
    Infer the ColSpec type from a type hint.
    The inferred dtype should be one of the supported data types in COLSPEC_TYPES.
    """
    if type_hint == Any:
        color_warning(
            message="Any type hint is inferred as AnyType, and MLflow doesn't validate the data "
            "for this type. Please use a more specific type hint to enable data validation.",
            stacklevel=2,
            color="yellow_bold",
        )
        return ColSpecType(dtype=AnyType(), required=True)
    if datatype := TYPE_HINTS_TO_DATATYPE_MAPPING.get(type_hint):
        return ColSpecType(dtype=datatype, required=True)
    elif _is_pydantic_type_hint(type_hint):
        dtype = _infer_type_from_pydantic_model(type_hint)
        return ColSpecType(dtype=dtype, required=True)
    elif origin_type := get_origin(type_hint):
        args = get_args(type_hint)
        if origin_type is list:
            internal_type = _get_element_type_of_list_type_hint(type_hint)
            return ColSpecType(
                dtype=Array(_infer_colspec_type_from_type_hint(type_hint=internal_type).dtype),
                required=True,
            )
        if origin_type is dict:
            if len(args) == 2:
                if args[0] != str:
                    raise InvalidTypeHintException(
                        message=f"Dictionary key type must be str, got {args[0]} in type hint "
                        f"{_type_hint_repr(type_hint)}"
                    )
                return ColSpecType(
                    dtype=Map(_infer_colspec_type_from_type_hint(type_hint=args[1]).dtype),
                    required=True,
                )
            raise InvalidTypeHintException(
                message="Dictionary type hint must contain two element types, got "
                f"{_type_hint_repr(type_hint)}"
            )
        if origin_type in UNION_TYPES:
            if NONE_TYPE in args:
                # This case shouldn't happen, but added for completeness
                if len(args) < 2:
                    raise InvalidTypeHintException(
                        message=f"Union type hint must contain at least one non-None type, "
                        f"got {_type_hint_repr(type_hint)}"
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
                        "and MLflow doesn't validate the data against its element types."
                    )
                    return ColSpecType(dtype=AnyType(), required=False)
            # Union type with all valid types is matched as AnyType
            else:
                _logger.warning(
                    "Union type hint is inferred as AnyType, and MLflow doesn't validate the data "
                    "against its element types."
                )
                return ColSpecType(dtype=AnyType(), required=True)
    _raise_type_hint_error(type_hint)


def _raise_type_hint_error(type_hint: type[Any]) -> None:
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
            message=f"Invalid type hint `{_type_hint_repr(type_hint)}`, it must include "
            f"a valid element type. {SUPPORTED_TYPE_HINT_MSG}"
        )
    raise UnsupportedTypeHintException(type_hint=type_hint)


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
            raise InvalidTypeHintException(
                message=f"Optional field `{field_name}` in Pydantic model `{model.__name__}` "
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
        raise InvalidTypeHintException(
            message="The following fields in the Pydantic model do not have type annotations: "
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
    return model.model_fields


def model_validate(model: pydantic.BaseModel, values: Any) -> None:
    # use strict mode to avoid any data conversion here
    # e.g. "123" will not be converted to 123 if the type is int
    model.model_validate(values, strict=True)


def field_required(field: type[FIELD_TYPE]) -> bool:
    return field.is_required()


def _get_element_type_of_list_type_hint(type_hint: type[list[Any]]) -> Any:
    """
    Get the element type of list[...] type hint
    """
    args = get_args(type_hint)
    # Optional[list[...]]
    if type(None) in args:
        raise MlflowException.invalid_parameter_value(OPTIONAL_INPUT_MSG)
    # a valid list[...] type hint must only contain one argument
    if len(args) == 0:
        raise InvalidTypeHintException(
            message=f"Type hint `{_type_hint_repr(type_hint)}` doesn't contain a collection "
            "element type. Fix by adding an element type to the collection type definition, "
            "e.g. `list[str]` instead of `list`."
        )
    if len(args) > 1:
        raise InvalidTypeHintException(
            message=f"Type hint `{_type_hint_repr(type_hint)}` contains {len(args)} element types. "
            "Collections must have only a single type definition e.g. `list[int]` is valid; "
            "`list[str, int]` is invalid."
        )
    return args[0]


def _is_list_type_hint(type_hint: type[Any]) -> bool:
    origin_type = _get_origin_type(type_hint)
    return type_hint == list or origin_type is list


def _infer_schema_from_list_type_hint(type_hint: type[list[Any]]) -> Schema:
    """
    Infer schema from a list type hint.
    The type hint must be list[...], and the inferred schema contains a
    single ColSpec, where the type is based on the element type of the list type hint,
    since ColSpec represents a column's data type of the dataset.
    e.g. list[int] -> Schema([ColSpec(type=DataType.long, required=True)])
    A valid `predict` function of a pyfunc model must use list type hint for the input.
    """
    if not _is_list_type_hint(type_hint):
        # This should be invalid, but to keep backwards compatibility of ChatCompletionRequest
        # type hint used in some rag models, we raise UnsupportedTypeHintException here
        # so that the model with such type hint can still be logged
        raise MlflowException.invalid_parameter_value(
            message="Type hints must be wrapped in list[...] because MLflow assumes the "
            "predict method to take multiple input instances. Specify your type hint as "
            f"`list[{_type_hint_repr(type_hint)}]` for a valid signature."
        )
    internal_type = _get_element_type_of_list_type_hint(type_hint)
    return _infer_schema_from_type_hint(internal_type)


def _infer_schema_from_type_hint(type_hint: type[Any]) -> Schema:
    col_spec_type = _infer_colspec_type_from_type_hint(type_hint)
    # Creating Schema with unnamed optional inputs is not supported
    if col_spec_type.required is False:
        raise InvalidTypeHintException(message=OPTIONAL_INPUT_MSG)
    return Schema([ColSpec(type=col_spec_type.dtype, required=col_spec_type.required)])


def _validate_data_against_type_hint(data: Any, type_hint: type[Any]) -> Any:
    """
    Validate the data against provided type hint.
    The allowed conversions are:
        dictionary data with Pydantic model type hint -> Pydantic model instance

    Args:
        data: The data to validate
        type_hint: The type hint to validate against
    """
    if _is_pydantic_type_hint(type_hint):
        # if data is a pydantic model instance, convert it to a dictionary for validation
        if isinstance(data, pydantic.BaseModel):
            data_dict = data.model_dump()
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise MlflowException.invalid_parameter_value(
                "Expecting example to be a dictionary or pydantic model instance for "
                f"Pydantic type hint, got {type(data)}"
            )
        try:
            model_validate(type_hint, data_dict)
        except pydantic.ValidationError as e:
            raise MlflowException.invalid_parameter_value(
                message=f"Data doesn't match type hint, error: {e}. Expected fields in the "
                f"type hint: {model_fields(type_hint)}; passed data: {data_dict}. Check "
                "https://mlflow.org/docs/latest/model/python_model.html#pydantic-model-type-hints-data-conversion"
                " for more details.",
            ) from e
        else:
            return type_hint(**data_dict) if isinstance(data, dict) else data
    elif type_hint == Any:
        return data
    elif type_hint in TYPE_HINTS_TO_DATATYPE_MAPPING:
        if _MLFLOW_IS_IN_SERVING_ENVIRONMENT.get():
            data = _parse_data_for_datatype_hint(data=data, type_hint=type_hint)
        if isinstance(data, type_hint):
            return data
        raise MlflowException.invalid_parameter_value(
            f"Expected type {_type_hint_repr(type_hint)}, but got {type(data).__name__}"
        )
    elif origin_type := get_origin(type_hint):
        args = get_args(type_hint)
        if origin_type is list:
            return _validate_list_elements(element_type=args[0], data=data)
        elif origin_type is dict:
            return _validate_dict_elements(element_type=args[1], data=data)
        elif origin_type in UNION_TYPES:
            # Optional type
            if NONE_TYPE in args:
                if data is None:
                    return data
                if len(args) == 2:
                    effective_type = next((arg for arg in args if arg is not NONE_TYPE), None)
                    return _validate_data_against_type_hint(data=data, type_hint=effective_type)
            # Union type with all valid types is matched as AnyType
            # no validation needed for AnyType
            return data
    _raise_type_hint_error(type_hint)


def _parse_data_for_datatype_hint(data: Any, type_hint: type[Any]) -> Any:
    """
    Parse the data based on the type hint.
    This should only be used in MLflow serving environment to convert
    json data to the expected format.
    Allowed conversions:
        - string data with datetime type hint -> datetime object
        - string data with bytes type hint -> bytes object
    """
    if type_hint == bytes and isinstance(data, str):
        # The assumption is that the data is base64 encoded, and
        # scoring server accepts base64 encoded string for bytes fields.
        # MLflow uses the same method for saving input example
        # via base64.encodebytes(x).decode("ascii")
        return base64.decodebytes(bytes(data, "utf8"))
    if type_hint == datetime and isinstance(data, str):
        # The assumption is that the data is in ISO format
        return datetime.fromisoformat(data)
    return data


class ValidationResult(NamedTuple):
    value: Any | None = None
    error_message: str | None = None


def _get_data_validation_result(data: Any, type_hint: type[Any]) -> ValidationResult:
    try:
        value = _validate_data_against_type_hint(data=data, type_hint=type_hint)
        return ValidationResult(value=value)
    except MlflowException as e:
        return ValidationResult(error_message=e.message)


def _type_hint_repr(type_hint: type[Any]) -> str:
    return (
        type_hint.__name__
        if _is_pydantic_type_hint(type_hint) or type(type_hint) == type
        else str(type_hint)
    )


def _validate_list_elements(element_type: type[Any], data: Any) -> list[Any]:
    if not isinstance(data, list):
        raise MlflowException.invalid_parameter_value(
            f"Expected list, but got {type(data).__name__}"
        )
    invalid_elems = []
    result = []
    for elem in data:
        validation_result = _get_data_validation_result(data=elem, type_hint=element_type)
        if validation_result.error_message:
            invalid_elems.append((str(elem), validation_result.error_message))
        else:
            result.append(validation_result.value)
    if invalid_elems:
        invalid_elems_msg = (
            f"{invalid_elems[:5]} ... (truncated)" if len(invalid_elems) > 5 else invalid_elems
        )
        raise MlflowException.invalid_parameter_value(
            f"Failed to validate data against type hint `list[{_type_hint_repr(element_type)}]`, "
            f"invalid elements: {invalid_elems_msg}"
        )
    return result


def _validate_dict_elements(element_type: type[Any], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise MlflowException.invalid_parameter_value(
            f"Expected dict, but got {type(data).__name__}"
        )
    invalid_elems = {}
    result = {}
    for key, value in data.items():
        if not isinstance(key, str):
            invalid_elems[str(key)] = f"Key must be a string, got {type(key).__name__}"
            continue
        validation_result = _get_data_validation_result(data=value, type_hint=element_type)
        if validation_result.error_message:
            invalid_elems[key] = validation_result.error_message
        else:
            result[key] = validation_result.value
    if invalid_elems:
        raise MlflowException.invalid_parameter_value(
            f"Failed to validate data against type hint "
            f"`dict[str, {_type_hint_repr(element_type)}]`, "
            f"invalid elements: {invalid_elems}"
        )
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
    This function is used in data validation of @pyfunc to support compatibility with
    functions such as mlflow.evaluate and spark_udf since they accept pandas DF as input.
    NB: the input pandas DataFrame must contain a single column with the same type as the type hint.
    Supported conversions:
        - pandas DataFrame with a single column + list[...] type hint -> list
        - pandas DataFrame with multiple columns + list[dict[...]] type hint -> list[dict[...]]
    """
    import pandas as pd

    result = data
    if isinstance(data, pd.DataFrame) and type_hint != pd.DataFrame:
        origin_type = _get_origin_type(type_hint)
        if origin_type is not list:
            raise MlflowException(
                "Only `list[...]` type hint supports pandas DataFrame input "
                f"with a single column. But got {_type_hint_repr(type_hint)}."
            )
        element_type = _get_element_type_of_list_type_hint(type_hint)
        # This is needed for list[dict] or list[pydantic.BaseModel] type hints
        # since the data can be converted to pandas DataFrame with multiple columns
        # inside spark_udf
        if element_type is dict or _is_pydantic_type_hint(element_type):
            # if the column is 0, then each row is a dictionary
            if list(data.columns) == [0]:
                result = data.iloc[:, 0].tolist()
            else:
                result = data.to_dict(orient="records")
        else:
            if len(data.columns) != 1:
                # TODO: remove the warning and raise Exception once the bug about evaluate
                # DF containing multiple columns is fixed
                _logger.warning(
                    "`predict` function with list[...] type hints of non-dictionary collection "
                    "type only supports pandas DataFrame with a single column. But got "
                    f"{len(data.columns)} columns. The data will be converted to a list "
                    "of the first column."
                )
            result = data.iloc[:, 0].tolist()
        # only sanitize the data when it's converted from pandas DataFrame
        # since spark_udf implicitly converts lists into numpy arrays
        return _sanitize_data(result)

    return data


def _sanitize_data(data: Any) -> Any:
    """
    Sanitize the data by converting any numpy lists to Python lists.
    This is needed because spark_udf (pandas_udf) implicitly converts lists into numpy arrays.

    For example, below udf demonstrates the behavior:
        df = spark.createDataFrame(pd.DataFrame({"input": [["a", "b"], ["c", "d"]]}))
        @pandas_udf(ArrayType(StringType()))
        def my_udf(input_series: pd.Series) -> pd.Series:
            print(type(input_series.iloc[0]))
        df.withColumn("output", my_udf("input")).show()
    """
    if np := _try_import_numpy():
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, list):
            data = [_sanitize_data(elem) for elem in data]
        if isinstance(data, dict):
            data = {key: _sanitize_data(value) for key, value in data.items()}
        if isinstance(data, float) and np.isnan(data):
            data = None
    return data
