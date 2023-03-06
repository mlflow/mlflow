import json
from enum import Enum

import numpy as np
import string
from typing import Dict, Any, List, Union, Optional

from mlflow.exceptions import MlflowException


class DataType(Enum):
    """
    MLflow data types.
    """

    def __new__(cls, value, numpy_type, spark_type, pandas_type=None):
        res = object.__new__(cls)
        res._value_ = value
        res._numpy_type = numpy_type
        res._spark_type = spark_type
        res._pandas_type = pandas_type if pandas_type is not None else numpy_type
        return res

    # NB: We only use pandas extension type for strings. There are also pandas extension types for
    # integers and boolean values. We do not use them here for now as most downstream tools are
    # most likely to use / expect native numpy types and would not be compatible with the extension
    # types.
    boolean = (1, np.dtype("bool"), "BooleanType")
    """Logical data (True, False) ."""
    integer = (2, np.dtype("int32"), "IntegerType")
    """32b signed integer numbers."""
    long = (3, np.dtype("int64"), "LongType")
    """64b signed integer numbers. """
    float = (4, np.dtype("float32"), "FloatType")
    """32b floating point numbers. """
    double = (5, np.dtype("float64"), "DoubleType")
    """64b floating point numbers. """
    string = (6, np.dtype("str"), "StringType", object)
    """Text data."""
    binary = (7, np.dtype("bytes"), "BinaryType", object)
    """Sequence of raw bytes."""
    datetime = (8, np.dtype("datetime64[ns]"), "TimestampType")
    """64b datetime data."""

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

    @classmethod
    def get_spark_types(cls):
        return [dt.to_spark() for dt in cls._member_map_.values()]


class ColSpec:
    """
    Specification of name and type of a single column in a dataset.
    """

    def __init__(
        self, type: DataType, name: Optional[str] = None  # pylint: disable=redefined-builtin
    ):
        self._name = name
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException(
                "Unsupported type '{}', expected instance of DataType or "
                "one of {}".format(type, [t.name for t in DataType])
            )
        if not isinstance(self.type, DataType):
            raise TypeError(
                "Expected mlflow.models.signature.Datatype or str for the 'type' "
                "argument, but got {}".format(self.type.__class__)
            )

    @property
    def type(self) -> DataType:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    def to_dict(self) -> Dict[str, Any]:
        if self.name is None:
            return {"type": self.type.name}
        else:
            return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        if isinstance(other, ColSpec):
            names_eq = (self.name is None and other.name is None) or self.name == other.name
            return names_eq and self.type == other.type
        return False

    def __repr__(self) -> str:
        if self.name is None:
            return repr(self.type)
        else:
            return "{name}: {type}".format(name=repr(self.name), type=repr(self.type))


class TensorInfo:
    """
    Representation of the shape and type of a Tensor.
    """

    def __init__(self, dtype: np.dtype, shape: Union[tuple, list]):
        if not isinstance(dtype, np.dtype):
            raise TypeError(
                "Expected `type` to be instance of `{}`, received `{}`".format(
                    np.dtype, type.__class__
                )
            )
        # Throw if size information exists flexible numpy data types
        if dtype.char in ["U", "S"] and not dtype.name.isalpha():
            raise MlflowException(
                "MLflow does not support size information in flexible numpy data types. Use"
                ' np.dtype("{}") instead'.format(dtype.name.rstrip(string.digits))
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
        return "Tensor({type}, {shape})".format(type=repr(self.dtype.name), shape=repr(self.shape))


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
            return "{name}: {info}".format(name=repr(self.name), info=repr(self._tensorInfo))


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
        if not (all(x.name is None for x in inputs) or all(x.name is not None for x in inputs)):
            raise MlflowException(
                "Creating Schema with a combination of named and unnamed inputs "
                "is not allowed. Got input names {}".format([x.name for x in inputs])
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
        self._inputs = inputs

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

    def has_input_names(self) -> bool:
        """Return true iff this schema declares names, false otherwise."""
        return self.inputs and self.inputs[0].name is not None

    def input_types(self) -> List[Union[DataType, np.dtype]]:
        """Get types of the represented dataset."""
        return [x.type for x in self.inputs]

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
        from pyspark.sql.types import StructType, StructField

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
