import builtins
import datetime as dt
import importlib.util
import json
import string
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
from PIL import Image

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental


class DataType(Enum):
    """
    MLflow data types.
    """

    def __new__(cls, value, numpy_type, spark_type, pandas_type=None, python_type=None):
        res = object.__new__(cls)
        res._value_ = value
        res._numpy_type = numpy_type
        res._spark_type = spark_type
        res._pandas_type = pandas_type if pandas_type is not None else numpy_type
        res._python_type = python_type if python_type is not None else numpy_type
        return res

    # NB: We only use pandas extension type for strings. There are also pandas extension types for
    # integers and boolean values. We do not use them here for now as most downstream tools are
    # most likely to use / expect native numpy types and would not be compatible with the extension
    # types.
    boolean = (1, np.dtype("bool"), "BooleanType", np.dtype("bool"), bool)
    """Logical data (True, False) ."""
    integer = (2, np.dtype("int32"), "IntegerType", np.dtype("int32"), int)
    """32b signed integer numbers."""
    long = (3, np.dtype("int64"), "LongType", np.dtype("int64"), int)
    """64b signed integer numbers. """
    float = (4, np.dtype("float32"), "FloatType", np.dtype("float32"), builtins.float)
    """32b floating point numbers. """
    double = (5, np.dtype("float64"), "DoubleType", np.dtype("float64"), builtins.float)
    """64b floating point numbers. """
    string = (6, np.dtype("str"), "StringType", object, str)
    """Text data."""
    binary = (7, np.dtype("bytes"), "BinaryType", object, bytes)
    """Sequence of raw bytes."""
    datetime = (
        8,
        np.dtype("datetime64[ns]"),
        "TimestampType",
        np.dtype("datetime64[ns]"),
        dt.date,
    )
    """64b datetime data."""
    pilimage = (9, "PILImage", "PILImage", object, Image.Image)
    """pil image."""

    def __repr__(self):
        return self.name

    def to_numpy(self) -> np.dtype:
        """Get equivalent numpy data type."""
        return self._numpy_type

    def to_pandas(self) -> np.dtype:
        """Get equivalent pandas data type."""
        return self._pandas_type

    def to_spark(self):
        import pyspark.sql.types

        return getattr(pyspark.sql.types, self._spark_type)()

    def to_python(self):
        """Get equivalent python data type."""
        return self._python_type

    @classmethod
    def is_boolean(cls, value):
        return type(value) in DataType.boolean.get_all_types()

    @classmethod
    def is_integer(cls, value):
        return type(value) in DataType.integer.get_all_types()

    @classmethod
    def is_long(cls, value):
        return type(value) in DataType.long.get_all_types()

    @classmethod
    def is_float(cls, value):
        return type(value) in DataType.float.get_all_types()

    @classmethod
    def is_double(cls, value):
        return type(value) in DataType.double.get_all_types()

    @classmethod
    def is_string(cls, value):
        return type(value) in DataType.string.get_all_types()

    @classmethod
    def is_binary(cls, value):
        return type(value) in DataType.binary.get_all_types()

    @classmethod
    def is_datetime(cls, value):
        return type(value) in DataType.datetime.get_all_types()

    def get_all_types(self):
        types = [self.to_numpy(), self.to_pandas(), self.to_python()]
        if importlib.util.find_spec("pyspark") is not None:
            types.append(self.to_spark())
        if self.name == "datetime":
            types.extend([np.datetime64, dt.datetime])
        return types

    @classmethod
    def get_spark_types(cls):
        return [dt.to_spark() for dt in cls._member_map_.values()]

    @classmethod
    def from_numpy_type(cls, np_type):
        return next((v for v in cls._member_map_.values() if v.to_numpy() == np_type), None)


class ColSpec:
    """
    Specification of name and type of a single column in a dataset.
    """

    def __init__(
        self,
        type: Union[DataType, str],  # pylint: disable=redefined-builtin
        name: Optional[str] = None,
        optional: bool = False,
    ):
        self._name = name
        self._optional = optional
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException(
                f"Unsupported type '{type}', expected instance of DataType or "
                f"one of {[t.name for t in DataType]}"
            )
        if not isinstance(self.type, DataType):
            raise TypeError(
                "Expected mlflow.models.signature.Datatype or str for the 'type' "
                f"argument, but got {self.type.__class__}"
            )

    @property
    def type(self) -> DataType:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    @experimental
    @property
    def optional(self) -> bool:
        """Whether this column is optional."""
        return self._optional

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type.name}
        if self.name is not None:
            d["name"] = self.name
        if self.optional:
            d["optional"] = self.optional
        return d

    def __eq__(self, other) -> bool:
        if isinstance(other, ColSpec):
            names_eq = (self.name is None and other.name is None) or self.name == other.name
            return names_eq and self.type == other.type and self.optional == other.optional
        return False

    def __repr__(self) -> str:
        if self.name is None:
            return repr(self.type)
        else:
            return "{name}: {type}{optional}".format(
                name=repr(self.name),
                type=repr(self.type),
                optional=" (optional)" if self.optional else "",
            )


class TensorInfo:
    """
    Representation of the shape and type of a Tensor.
    """

    def __init__(self, dtype: np.dtype, shape: Union[tuple, list]):
        if not isinstance(dtype, np.dtype):
            raise TypeError(
                f"Expected `type` to be instance of `{np.dtype}`, received `{ type.__class__}`"
            )
        # Throw if size information exists flexible numpy data types
        if dtype.char in ["U", "S"] and not dtype.name.isalpha():
            raise MlflowException(
                "MLflow does not support size information in flexible numpy data types. Use"
                f' np.dtype("{dtype.name.rstrip(string.digits)}") instead'
            )

        if not isinstance(shape, (tuple, list)):
            raise TypeError(
                "Expected `shape` to be instance of `{}` or `{}`, received `{}`".format(
                    tuple, list, shape.__class__
                )
            )
        self._dtype = dtype
        self._shape = tuple(shape)

    @property
    def dtype(self) -> np.dtype:
        """
        A unique character code for each of the 21 different numpy built-in types.
        See https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype for details.
        """
        return self._dtype

    @property
    def shape(self) -> tuple:
        """The tensor shape"""
        return self._shape

    def to_dict(self) -> Dict[str, Any]:
        return {"dtype": self._dtype.name, "shape": self._shape}

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `dtype` and `shape` keys.
        """
        if not {"dtype", "shape"} <= set(kwargs.keys()):
            raise MlflowException(
                "Missing keys in TensorSpec JSON. Expected to find keys `dtype` and `shape`"
            )
        tensor_type = np.dtype(kwargs["dtype"])
        tensor_shape = tuple(kwargs["shape"])
        return cls(tensor_type, tensor_shape)

    def __repr__(self) -> str:
        return f"Tensor({self.dtype.name!r}, {self.shape!r})"


class TensorSpec:
    """
    Specification used to represent a dataset stored as a Tensor.
    """

    def __init__(
        self,
        type: np.dtype,  # pylint: disable=redefined-builtin
        shape: Union[tuple, list],
        name: Optional[str] = None,
    ):
        self._name = name
        self._tensorInfo = TensorInfo(type, shape)

    @property
    def type(self) -> np.dtype:
        """
        A unique character code for each of the 21 different numpy built-in types.
        See https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype for details.
        """
        return self._tensorInfo.dtype

    @property
    def name(self) -> Optional[str]:
        """The tensor name or None if the tensor is unnamed."""
        return self._name

    @property
    def shape(self) -> tuple:
        """The tensor shape"""
        return self._tensorInfo.shape

    @experimental
    @property
    def optional(self) -> bool:
        """Whether this tensor is optional."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        if self.name is None:
            return {"type": "tensor", "tensor-spec": self._tensorInfo.to_dict()}
        else:
            return {"name": self.name, "type": "tensor", "tensor-spec": self._tensorInfo.to_dict()}

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and `tensor-spec` keys.
        """
        if not {"tensor-spec", "type"} <= set(kwargs.keys()):
            raise MlflowException(
                "Missing keys in TensorSpec JSON. Expected to find keys `tensor-spec` and `type`"
            )
        if kwargs["type"] != "tensor":
            raise MlflowException("Type mismatch, TensorSpec expects `tensor` as the type")
        tensor_info = TensorInfo.from_json_dict(**kwargs["tensor-spec"])
        return cls(
            tensor_info.dtype, tensor_info.shape, kwargs["name"] if "name" in kwargs else None
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, TensorSpec):
            names_eq = (self.name is None and other.name is None) or self.name == other.name
            return names_eq and self.type == other.type and self.shape == other.shape
        return False

    def __repr__(self) -> str:
        if self.name is None:
            return repr(self._tensorInfo)
        else:
            return f"{self.name!r}: {self._tensorInfo!r}"


class Schema:
    """
    Specification of a dataset.

    Schema is represented as a list of :py:class:`ColSpec` or :py:class:`TensorSpec`. A combination
    of `ColSpec` and `TensorSpec` is not allowed.

    The dataset represented by a schema can be named, with unique non empty names for every input.
    In the case of :py:class:`ColSpec`, the dataset columns can be unnamed with implicit integer
    index defined by their list indices.
    Combination of named and unnamed data inputs are not allowed.
    """

    def __init__(self, inputs: List[Union[ColSpec, TensorSpec]]):
        if not isinstance(inputs, list):
            raise MlflowException.invalid_parameter_value(
                f"Inputs of Schema must be a list, got type {type(inputs).__name__}"
            )
        if not inputs:
            raise MlflowException.invalid_parameter_value(
                "Creating Schema with empty inputs is not allowed."
            )
        if not (all(x.name is None for x in inputs) or all(x.name is not None for x in inputs)):
            raise MlflowException(
                "Creating Schema with a combination of named and unnamed inputs "
                f"is not allowed. Got input names {[x.name for x in inputs]}"
            )
        if not (
            all(isinstance(x, TensorSpec) for x in inputs)
            or all(isinstance(x, ColSpec) for x in inputs)
        ):
            raise MlflowException(
                "Creating Schema with a combination of {0} and {1} is not supported. "
                "Please choose one of {0} or {1}".format(ColSpec.__class__, TensorSpec.__class__)
            )
        if (
            all(isinstance(x, TensorSpec) for x in inputs)
            and len(inputs) > 1
            and any(x.name is None for x in inputs)
        ):
            raise MlflowException(
                "Creating Schema with multiple unnamed TensorSpecs is not supported. "
                "Please provide names for each TensorSpec."
            )
        if all(x.name is None for x in inputs) and any(x.optional is True for x in inputs):
            raise MlflowException(
                "Creating Schema with unnamed optional inputs is not supported. "
                "Please name all inputs or make all inputs required."
            )
        self._inputs = inputs

    def __len__(self):
        return len(self._inputs)

    def __iter__(self):
        return iter(self._inputs)

    @property
    def inputs(self) -> List[Union[ColSpec, TensorSpec]]:
        """Representation of a dataset that defines this schema."""
        return self._inputs

    def is_tensor_spec(self) -> bool:
        """Return true iff this schema is specified using TensorSpec"""
        return self.inputs and isinstance(self.inputs[0], TensorSpec)

    def input_names(self) -> List[Union[str, int]]:
        """Get list of data names or range of indices if the schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs)]

    def required_input_names(self) -> List[Union[str, int]]:
        """Get list of required data names or range of indices if schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs) if not x.optional]

    @experimental
    def optional_input_names(self) -> List[Union[str, int]]:
        """Get list of optional data names or range of indices if schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs) if x.optional]

    def has_input_names(self) -> bool:
        """Return true iff this schema declares names, false otherwise."""
        return self.inputs and self.inputs[0].name is not None

    def input_types(self) -> List[Union[DataType, np.dtype]]:
        """Get types for each column in the schema."""
        return [x.type for x in self.inputs]

    def input_types_dict(self) -> Dict[str, Union[DataType, np.dtype]]:
        """Maps column names to types, iff this schema declares names."""
        if not self.has_input_names():
            raise MlflowException("Cannot get input types as a dict for schema without names.")
        return {x.name: x.type for x in self.inputs}

    def numpy_types(self) -> List[np.dtype]:
        """Convenience shortcut to get the datatypes as numpy types."""
        if self.is_tensor_spec():
            return [x.type for x in self.inputs]
        return [x.type.to_numpy() for x in self.inputs]

    def pandas_types(self) -> List[np.dtype]:
        """Convenience shortcut to get the datatypes as pandas types. Unsupported by TensorSpec."""
        if self.is_tensor_spec():
            raise MlflowException("TensorSpec only supports numpy types, use numpy_types() instead")
        return [x.type.to_pandas() for x in self.inputs]

    def as_spark_schema(self):
        """Convert to Spark schema. If this schema is a single unnamed column, it is converted
        directly the corresponding spark data type, otherwise it's returned as a struct (missing
        column names are filled with an integer sequence).
        Unsupported by TensorSpec.
        """
        if self.is_tensor_spec():
            raise MlflowException("TensorSpec cannot be converted to spark dataframe")
        if len(self.inputs) == 1 and self.inputs[0].name is None:
            return self.inputs[0].type.to_spark()
        from pyspark.sql.types import StructField, StructType

        return StructType(
            [
                StructField(name=col.name or str(i), dataType=col.type.to_spark())
                for i, col in enumerate(self.inputs)
            ]
        )

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps([x.to_dict() for x in self.inputs])

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.inputs]

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from a json string."""

        def read_input(x: dict):
            return TensorSpec.from_json_dict(**x) if x["type"] == "tensor" else ColSpec(**x)

        return cls([read_input(x) for x in json.loads(json_str)])

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return self.inputs == other.inputs
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.inputs)


@experimental
class ParamSpec:
    """
    Specification used to represent parameters for the model.
    """

    def __init__(
        self,
        name: str,
        dtype: Union[DataType, str],
        default: Union[DataType, List[DataType], None],
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self._name = str(name)
        self._shape = tuple(shape) if shape is not None else None

        try:
            self._dtype = DataType[dtype] if isinstance(dtype, str) else dtype
        except KeyError:
            supported_types = [t.name for t in DataType if t.name != "binary"]
            raise MlflowException.invalid_parameter_value(
                f"Unsupported type '{dtype}', expected instance of DataType or "
                f"one of {supported_types}",
            )
        if not isinstance(self.dtype, DataType):
            raise TypeError(
                "Expected mlflow.models.signature.Datatype or str for the 'dtype' "
                f"argument, but got {self.dtype.__class__}"
            )
        if self.dtype == DataType.binary:
            raise MlflowException.invalid_parameter_value(
                f"Binary type is not supported for parameters, ParamSpec '{self.name}'"
                "has dtype 'binary'",
            )

        # This line makes sure repr(self) works fine
        self._default = default
        self._default = self.validate_type_and_shape(repr(self), default, self.dtype, self.shape)

    @classmethod
    def validate_param_spec(
        cls, value: Union[DataType, List[DataType], None], param_spec: "ParamSpec"
    ):
        return cls.validate_type_and_shape(
            repr(param_spec), value, param_spec.dtype, param_spec.shape
        )

    @classmethod
    def enforce_param_datatype(cls, name, value, dtype: DataType):
        """
        Enforce the value matches the data type.

        The following type conversions are allowed:

        1. int -> long, float, double
        2. long -> float, double
        3. float -> double
        4. any -> datetime (try conversion)

        Any other type mismatch will raise error.

        :param name: parameter name
        :param value: parameter value
        :param t: expected data type
        """
        if value is None:
            return

        if dtype == DataType.datetime:
            try:
                datetime_value = np.datetime64(value).item()
                if isinstance(datetime_value, int):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid value for param {name}, it should "
                        f"be convertible to datetime.date/datetime, got {value}"
                    )
                return datetime_value
            except ValueError as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to convert value {value} from type {type(value).__name__} "
                    f"to {dtype} for param {name}"
                ) from e

        # Note that np.isscalar(datetime.date(...)) is False
        if not np.isscalar(value):
            raise MlflowException.invalid_parameter_value(
                f"Value should be a scalar for param {name}, got {value}"
            )

        # Always convert to python native type for params
        if getattr(DataType, f"is_{dtype.name}")(value):
            return DataType[dtype.name].to_python()(value)

        if (
            (
                DataType.is_integer(value)
                and dtype in (DataType.long, DataType.float, DataType.double)
            )
            or (DataType.is_long(value) and dtype in (DataType.float, DataType.double))
            or (DataType.is_float(value) and dtype == DataType.double)
        ):
            try:
                return DataType[dtype.name].to_python()(value)
            except ValueError as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to convert value {value} from type {type(value).__name__} "
                    f"to {dtype} for param {name}"
                ) from e

        raise MlflowException.invalid_parameter_value(
            f"Incompatible types for param {name}. Can not safely convert {type(value).__name__} "
            f"to {dtype}.",
        )

    @classmethod
    def validate_type_and_shape(
        cls,
        spec: str,
        value: Union[DataType, List[DataType], None],
        value_type: DataType,
        shape: Optional[Tuple[int, ...]],
    ):
        """
        Validate that the value has the expected type and shape.
        """

        def _is_1d_array(value):
            return isinstance(value, (list, np.ndarray)) and np.array(value).ndim == 1

        if shape is None:
            return cls.enforce_param_datatype(f"{spec} with shape None", value, value_type)
        elif shape == (-1,):
            if not _is_1d_array(value):
                raise MlflowException.invalid_parameter_value(
                    f"Value must be a 1D array with shape (-1,) for param {spec}, "
                    f"received {type(value).__name__} with ndim {np.array(value).ndim}",
                )
            return [
                cls.enforce_param_datatype(f"{spec} internal values", v, value_type) for v in value
            ]
        else:
            raise MlflowException.invalid_parameter_value(
                "Shape must be None for scalar value or (-1,) for 1D array value "
                f"for ParamSpec {spec}), received {shape}",
            )

    @property
    def name(self) -> str:
        """The name of the parameter."""
        return self._name

    @property
    def dtype(self) -> DataType:
        """The parameter data type."""
        return self._dtype

    @property
    def default(self) -> Union[DataType, List[DataType], None]:
        """Default value of the parameter."""
        return self._default

    @property
    def shape(self) -> Optional[tuple]:
        """
        The parameter shape.
        If shape is None, the parameter is a scalar.
        """
        return self._shape

    class ParamSpecTypedDict(TypedDict):
        name: str
        type: str
        default: Union[DataType, List[DataType], None]
        shape: Optional[Tuple[int, ...]]

    def to_dict(self) -> ParamSpecTypedDict:
        if self.shape is None:
            default_value = (
                self.default.isoformat() if self.dtype.name == "datetime" else self.default
            )
        elif self.shape == (-1,):
            default_value = (
                [v.isoformat() for v in self.default]
                if self.dtype.name == "datetime"
                else self.default
            )
        return {
            "name": self.name,
            "type": self.dtype.name,
            "default": default_value,
            "shape": self.shape,
        }

    def __eq__(self, other) -> bool:
        if isinstance(other, ParamSpec):
            return (
                self.name == other.name
                and self.dtype == other.dtype
                and self.default == other.default
                and self.shape == other.shape
            )
        return False

    def __repr__(self) -> str:
        shape = f" (shape: {self.shape})" if self.shape is not None else ""
        return f"{self.name!r}: {self.dtype!r} (default: {self.default}){shape}"

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `name`, `type` and `default` keys.
        """
        # For backward compatibility, we accept both `type` and `dtype` keys
        required_keys1 = {"name", "dtype", "default"}
        required_keys2 = {"name", "type", "default"}

        if not (required_keys1.issubset(kwargs) or required_keys2.issubset(kwargs)):
            raise MlflowException.invalid_parameter_value(
                "Missing keys in ParamSpec JSON. Expected to find "
                "keys `name`, `type`(or `dtype`) and `default`. "
                f"Received keys: {kwargs.keys()}"
            )
        dtype = kwargs.get("type") or kwargs.get("dtype")
        return cls(
            name=str(kwargs["name"]),
            dtype=DataType[dtype],
            default=kwargs["default"],
            shape=kwargs.get("shape"),
        )


@experimental
class ParamSchema:
    """
    Specification of parameters applicable to the model.
    ParamSchema is represented as a list of :py:class:`ParamSpec`.
    """

    def __init__(self, params: List[ParamSpec]):
        if not all(isinstance(x, ParamSpec) for x in params):
            raise MlflowException.invalid_parameter_value(
                f"ParamSchema inputs only accept {ParamSchema.__class__}"
            )
        if duplicates := self._find_duplicates(params):
            raise MlflowException.invalid_parameter_value(
                f"Duplicated parameters found in schema: {duplicates}"
            )
        self._params = params

    @staticmethod
    def _find_duplicates(params: List[ParamSpec]) -> List[str]:
        param_names = [param_spec.name for param_spec in params]
        uniq_param = set()
        duplicates = []
        for name in param_names:
            if name in uniq_param:
                duplicates.append(name)
            else:
                uniq_param.add(name)
        return duplicates

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    @property
    def params(self) -> List[ParamSpec]:
        """Representation of ParamSchema as a list of ParamSpec."""
        return self._params

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from a json string."""
        return cls([ParamSpec.from_json_dict(**x) for x in json.loads(json_str)])

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.params]

    def __eq__(self, other) -> bool:
        if isinstance(other, ParamSchema):
            return self.params == other.params
        return False

    def __repr__(self) -> str:
        return repr(self.params)
