from __future__ import annotations

import builtins
import datetime as dt
import json
import string
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Optional, TypedDict, Union, get_args, get_origin

import numpy as np

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental

ARRAY_TYPE = "array"
OBJECT_TYPE = "object"
MAP_TYPE = "map"
ANY_TYPE = "any"
SPARKML_VECTOR_TYPE = "sparkml_vector"
ALLOWED_DTYPES = Union["Array", "DataType", "Map", "Object", "AnyType", str]
EXPECTED_TYPE_MESSAGE = (
    "Expected mlflow.types.schema.Datatype, mlflow.types.schema.Array, "
    "mlflow.types.schema.Object, mlflow.types.schema.Map, mlflow.types.schema.AnyType "
    "or str for the '{arg_name}' argument, but got {passed_type}"
)
COLSPEC_TYPES = Union["Array", "DataType", "Map", "Object", "AnyType"]

try:
    import pyspark  # noqa: F401

    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


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

    def __repr__(self):
        return self.name

    def to_numpy(self) -> np.dtype:
        """Get equivalent numpy data type."""
        return self._numpy_type

    def to_pandas(self) -> np.dtype:
        """Get equivalent pandas data type."""
        return self._pandas_type

    def to_spark(self):
        if self._spark_type == "VectorUDT":
            from pyspark.ml.linalg import VectorUDT

            return VectorUDT()
        else:
            import pyspark.sql.types

            return getattr(pyspark.sql.types, self._spark_type)()

    def to_python(self):
        """Get equivalent python data type."""
        return self._python_type

    @classmethod
    def check_type(cls, data_type, value):
        types = [data_type.to_numpy(), data_type.to_pandas(), data_type.to_python()]
        if data_type.name == "datetime":
            types.extend([np.datetime64, dt.datetime])
        if data_type.name == "binary":
            types.append(bytearray)
        if type(value) in types:
            return True
        if HAS_PYSPARK:
            return isinstance(value, type(data_type.to_spark()))
        return False

    @classmethod
    def all_types(cls):
        return list(DataType.__members__.values())

    @classmethod
    def get_spark_types(cls):
        return [dt.to_spark() for dt in cls._member_map_.values()]

    @classmethod
    def from_numpy_type(cls, np_type):
        return next((v for v in cls._member_map_.values() if v.to_numpy() == np_type), None)


class BaseType(ABC):
    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Determine if two objects are equal.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """
        The string representation of the object.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Dictionary representation of the object.
        """

    @abstractmethod
    def _merge(self, other: BaseType) -> BaseType:
        """
        Merge two objects and return the updated object if they're compatible.
        """


class Property(BaseType):
    """
    Specification used to represent a json-convertible object property.
    """

    def __init__(
        self,
        name: str,
        dtype: ALLOWED_DTYPES,
        required: bool = True,
    ) -> None:
        """
        Args:
            name: The name of the property
            dtype: The data type of the property
            required: Whether this property is required
        """
        if not isinstance(name, str):
            raise MlflowException.invalid_parameter_value(
                f"Expected name to be a string, got type {type(name).__name__}"
            )
        self._name = name
        try:
            self._dtype = DataType[dtype] if isinstance(dtype, str) else dtype
        except KeyError:
            raise MlflowException(
                f"Unsupported type '{dtype}', expected instance of DataType, Array, Object, Map or "
                f"one of {[t.name for t in DataType]}"
            )
        if not isinstance(self.dtype, (DataType, Array, Object, Map, AnyType)):
            raise MlflowException(
                EXPECTED_TYPE_MESSAGE.format(arg_name="dtype", passed_type=self.dtype)
            )
        self._required = required

    @property
    def name(self) -> str:
        """The property name."""
        return self._name

    @property
    def dtype(self) -> Union[DataType, "Array", "Object", "Map"]:
        """The property data type."""
        return self._dtype

    @property
    def required(self) -> bool:
        """Whether this property is required"""
        return self._required

    @required.setter
    def required(self, value: bool) -> None:
        self._required = value

    def __eq__(self, other) -> bool:
        if isinstance(other, Property):
            return (
                self.name == other.name
                and self.dtype == other.dtype
                and self.required == other.required
            )
        return False

    def __lt__(self, other) -> bool:
        return self.name < other.name

    def __repr__(self) -> str:
        required = "required" if self.required else "optional"
        return f"{self.name}: {self.dtype!r} ({required})"

    def to_dict(self):
        d = {"type": self.dtype.name} if isinstance(self.dtype, DataType) else self.dtype.to_dict()
        d["required"] = self.required
        return {self.name: d}

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain only one key as `name`, and
        the value should be a dictionary containing `type` and
        optional `required` keys.
        Example: {"property_name": {"type": "string", "required": True}}
        """
        if len(kwargs) != 1:
            raise MlflowException(
                f"Expected Property JSON to contain a single key as name, got {len(kwargs)} keys."
            )
        name, dic = kwargs.popitem()
        if not {"type"} <= set(dic.keys()):
            raise MlflowException(f"Missing keys in Property `{name}`. Expected to find key `type`")
        required = dic.pop("required", True)
        dtype = dic["type"]
        if dtype == ARRAY_TYPE:
            return cls(name=name, dtype=Array.from_json_dict(**dic), required=required)
        if dtype == SPARKML_VECTOR_TYPE:
            return SparkMLVector()
        if dtype == OBJECT_TYPE:
            return cls(name=name, dtype=Object.from_json_dict(**dic), required=required)
        if dtype == MAP_TYPE:
            return cls(name=name, dtype=Map.from_json_dict(**dic), required=required)
        if dtype == ANY_TYPE:
            return cls(name=name, dtype=AnyType(), required=required)
        return cls(name=name, dtype=dtype, required=required)

    def _merge(self, other: BaseType) -> Property:
        """
        Check if current property is compatible with another property and return
        the updated property.
        When two properties have the same name, we need to check if their dtypes
        are compatible or not.
        An example of two compatible properties:

            .. code-block:: python

                prop1 = Property(
                    name="a",
                    dtype=Object(
                        properties=[Property(name="a", dtype=DataType.string, required=False)]
                    ),
                )
                prop2 = Property(
                    name="a",
                    dtype=Object(
                        properties=[
                            Property(name="a", dtype=DataType.string),
                            Property(name="b", dtype=DataType.double),
                        ]
                    ),
                )
                merged_prop = prop1._merge(prop2)
                assert merged_prop == Property(
                    name="a",
                    dtype=Object(
                        properties=[
                            Property(name="a", dtype=DataType.string, required=False),
                            Property(name="b", dtype=DataType.double, required=False),
                        ]
                    ),
                )

        """
        if isinstance(other, AnyType):
            return Property(name=self.name, dtype=self.dtype, required=False)
        if not isinstance(other, Property):
            raise MlflowException(
                f"Can't merge property with non-property type: {type(other).__name__}"
            )
        if self.name != other.name:
            raise MlflowException("Can't merge properties with different names")
        required = self.required and other.required
        if isinstance(self.dtype, DataType) and isinstance(other.dtype, DataType):
            if self.dtype == other.dtype:
                return Property(name=self.name, dtype=self.dtype, required=required)
            raise MlflowException(f"Properties are incompatible for {self.dtype} and {other.dtype}")

        if isinstance(self.dtype, (Array, Object, Map, AnyType)):
            obj = self.dtype._merge(other.dtype)
            return Property(name=self.name, dtype=obj, required=required)

        raise MlflowException("Properties are incompatible")


class Object(BaseType):
    """
    Specification used to represent a json-convertible object.
    """

    def __init__(self, properties: list[Property]) -> None:
        self._check_properties(properties)
        # Sort by name to make sure the order is stable
        self._properties = sorted(properties)

    def _check_properties(self, properties):
        if not isinstance(properties, list):
            raise MlflowException.invalid_parameter_value(
                f"Expected properties to be a list, got type {type(properties).__name__}"
            )
        if len(properties) == 0:
            raise MlflowException.invalid_parameter_value(
                "Creating Object with empty properties is not allowed."
            )
        if any(not isinstance(v, Property) for v in properties):
            raise MlflowException.invalid_parameter_value(
                "Expected values to be instance of Property"
            )
        # check duplicated property names
        names = [prop.name for prop in properties]
        duplicates = {name for name in names if names.count(name) > 1}
        if len(duplicates) > 0:
            raise MlflowException.invalid_parameter_value(
                f"Found duplicated property names: {duplicates}"
            )

    @property
    def properties(self) -> list[Property]:
        """The list of object properties"""
        return self._properties

    @properties.setter
    def properties(self, value: list[Property]) -> None:
        self._check_properties(value)
        self._properties = sorted(value)

    def __eq__(self, other) -> bool:
        if isinstance(other, Object):
            return self.properties == other.properties
        return False

    def __repr__(self) -> str:
        joined = ", ".join(map(repr, self.properties))
        return "{" + joined + "}"

    def to_dict(self):
        properties = {
            name: value for prop in self.properties for name, value in prop.to_dict().items()
        }
        return {
            "type": OBJECT_TYPE,
            "properties": properties,
        }

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and
        `properties` keys.
        Example: {"type": "object", "properties": {"property_name": {"type": "string"}}}
        """
        if not {"properties", "type"} <= set(kwargs.keys()):
            raise MlflowException(
                "Missing keys in Object JSON. Expected to find keys `properties` and `type`"
            )
        if kwargs["type"] != OBJECT_TYPE:
            raise MlflowException("Type mismatch, Object expects `object` as the type")
        if not isinstance(kwargs["properties"], dict) or any(
            not isinstance(prop, dict) for prop in kwargs["properties"].values()
        ):
            raise MlflowException("Expected properties to be a dictionary of Property JSON")
        return cls(
            [Property.from_json_dict(**{name: prop}) for name, prop in kwargs["properties"].items()]
        )

    def _merge(self, other: BaseType) -> Object:
        """
        Check if the current object is compatible with another object and return
        the updated object.
        When we infer the signature from a list of objects, it is possible
        that one object has more properties than the other. In this case,
        we should mark those optional properties as required=False.
        For properties with the same name, we should check the compatibility
        of two properties and update.
        An example of two compatible objects:

            .. code-block:: python

                obj1 = Object(
                    properties=[
                        Property(name="a", dtype=DataType.string),
                        Property(name="b", dtype=DataType.double),
                    ]
                )
                obj2 = Object(
                    properties=[
                        Property(name="a", dtype=DataType.string),
                        Property(name="c", dtype=DataType.boolean),
                    ]
                )
                updated_obj = obj1._merge(obj2)
                assert updated_obj == Object(
                    properties=[
                        Property(name="a", dtype=DataType.string),
                        Property(name="b", dtype=DataType.double, required=False),
                        Property(name="c", dtype=DataType.boolean, required=False),
                    ]
                )

        """
        # Merging object type with AnyType makes all properties optional
        if isinstance(other, AnyType):
            return Object(
                properties=[
                    Property(name=prop.name, dtype=prop.dtype, required=False)
                    for prop in self.properties
                ]
            )
        if not isinstance(other, Object):
            raise MlflowException(
                f"Can't merge object with non-object type: {type(other).__name__}"
            )
        if self == other:
            return deepcopy(self)
        prop_dict1 = {prop.name: prop for prop in self.properties}
        prop_dict2 = {prop.name: prop for prop in other.properties}
        updated_properties = []
        # For each property in the first element, if it doesn't appear
        # later, we update required=False
        for k in prop_dict1.keys() - prop_dict2.keys():
            updated_properties.append(Property(name=k, dtype=prop_dict1[k].dtype, required=False))
        # For common keys, property type should be the same
        for k in prop_dict1.keys() & prop_dict2.keys():
            updated_properties.append(prop_dict1[k]._merge(prop_dict2[k]))
        # For each property appears in the second elements, if it doesn't
        # exist, we update and set required=False
        for k in prop_dict2.keys() - prop_dict1.keys():
            updated_properties.append(Property(name=k, dtype=prop_dict2[k].dtype, required=False))
        return Object(properties=updated_properties)


class Array(BaseType):
    """
    Specification used to represent a json-convertible array.
    """

    def __init__(
        self,
        dtype: ALLOWED_DTYPES,
    ) -> None:
        try:
            self._dtype = DataType[dtype] if isinstance(dtype, str) else dtype
        except KeyError:
            raise MlflowException(
                f"Unsupported type '{dtype}', expected instance of DataType, Array, Object, Map or "
                f"one of {[t.name for t in DataType]}"
            )
        if not isinstance(self.dtype, (Array, DataType, Object, Map, AnyType)):
            raise MlflowException(
                EXPECTED_TYPE_MESSAGE.format(arg_name="dtype", passed_type=self.dtype)
            )

    @property
    def dtype(self) -> Union["Array", DataType, Object, "Map", "AnyType"]:
        """The array data type."""
        return self._dtype

    def __eq__(self, other) -> bool:
        if isinstance(other, Array):
            return self.dtype == other.dtype
        return False

    def to_dict(self):
        items = (
            {"type": self.dtype.name} if isinstance(self.dtype, DataType) else self.dtype.to_dict()
        )
        return {"type": ARRAY_TYPE, "items": items}

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and
        `items` keys.
        Example: {"type": "array", "items": "string"}
        """
        if not {"items", "type"} <= set(kwargs.keys()):
            raise MlflowException(
                "Missing keys in Array JSON. Expected to find keys `items` and `type`"
            )
        if kwargs["type"] != ARRAY_TYPE:
            raise MlflowException("Type mismatch, Array expects `array` as the type")
        if not isinstance(kwargs["items"], dict):
            raise MlflowException("Expected items to be a dictionary of Object JSON")
        if not {"type"} <= set(kwargs["items"].keys()):
            raise MlflowException("Missing keys in Array's items JSON. Expected to find key `type`")

        if kwargs["items"]["type"] == OBJECT_TYPE:
            item_type = Object.from_json_dict(**kwargs["items"])
        elif kwargs["items"]["type"] == ARRAY_TYPE:
            item_type = Array.from_json_dict(**kwargs["items"])
        elif kwargs["items"]["type"] == SPARKML_VECTOR_TYPE:
            item_type = SparkMLVector()
        elif kwargs["items"]["type"] == MAP_TYPE:
            item_type = Map.from_json_dict(**kwargs["items"])
        elif kwargs["items"]["type"] == ANY_TYPE:
            item_type = AnyType()
        else:
            item_type = kwargs["items"]["type"]

        return cls(dtype=item_type)

    def __repr__(self) -> str:
        return f"Array({self.dtype!r})"

    def _merge(self, other: BaseType) -> Array:
        if isinstance(other, AnyType) or self == other:
            return deepcopy(self)
        if not isinstance(other, Array):
            raise MlflowException(f"Can't merge array with non-array type: {type(other).__name__}")
        if isinstance(self.dtype, DataType):
            if self.dtype == other.dtype:
                return Array(dtype=self.dtype)
            raise MlflowException(
                f"Array types are incompatible for {self} with dtype={self.dtype} and "
                f"{other} with dtype={other.dtype}"
            )

        if isinstance(self.dtype, (Array, Object, Map, AnyType)):
            return Array(dtype=self.dtype._merge(other.dtype))

        raise MlflowException(f"Array type {self!r} and {other!r} are incompatible")


class SparkMLVector(Array):
    """
    Specification used to represent a vector type in Spark ML.
    """

    def __init__(self):
        super().__init__(dtype=DataType.double)

    def to_dict(self):
        return {"type": SPARKML_VECTOR_TYPE}

    @classmethod
    def from_json_dict(cls, **kwargs):
        return SparkMLVector()

    def __repr__(self) -> str:
        return "SparkML vector"

    def __eq__(self, other) -> bool:
        return isinstance(other, SparkMLVector)

    def _merge(self, arr: BaseType) -> SparkMLVector:
        if isinstance(arr, SparkMLVector):
            return deepcopy(self)
        raise MlflowException("SparkML vector type can't be merged with another Array type.")


class Map(BaseType):
    """
    Specification used to represent a json-convertible map with string type keys.
    """

    def __init__(self, value_type: ALLOWED_DTYPES):
        try:
            self._value_type = DataType[value_type] if isinstance(value_type, str) else value_type
        except KeyError:
            raise MlflowException(
                f"Unsupported value type '{value_type}', expected instance of DataType, Array, "
                f"Object, Map or one of {[t.name for t in DataType]}"
            )
        if not isinstance(self._value_type, (Array, Map, DataType, Object, AnyType)):
            raise MlflowException.invalid_parameter_value(
                EXPECTED_TYPE_MESSAGE.format(arg_name="value_type", passed_type=self._value_type)
            )

    @property
    def value_type(self):
        return self._value_type

    def __repr__(self) -> str:
        return f"Map(str -> {self._value_type})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Map):
            return self.value_type == other.value_type
        return False

    def to_dict(self):
        values = (
            {"type": self.value_type.name}
            if isinstance(self.value_type, DataType)
            else self.value_type.to_dict()
        )
        return {"type": MAP_TYPE, "values": values}

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and
        `values` keys.
        Example: {"type": "map", "values": "string"}
        """
        if not {"values", "type"} <= set(kwargs.keys()):
            raise MlflowException(
                "Missing keys in Array JSON. Expected to find keys `items` and `type`"
            )
        if kwargs["type"] != MAP_TYPE:
            raise MlflowException("Type mismatch, Map expects `map` as the type")
        if not isinstance(kwargs["values"], dict):
            raise MlflowException("Expected values to be a dictionary of Object JSON")
        if not {"type"} <= set(kwargs["values"].keys()):
            raise MlflowException("Missing keys in Map's items JSON. Expected to find key `type`")
        if kwargs["values"]["type"] == OBJECT_TYPE:
            return cls(value_type=Object.from_json_dict(**kwargs["values"]))
        if kwargs["values"]["type"] == ARRAY_TYPE:
            return cls(value_type=Array.from_json_dict(**kwargs["values"]))
        if kwargs["values"]["type"] == SPARKML_VECTOR_TYPE:
            return SparkMLVector()
        if kwargs["values"]["type"] == MAP_TYPE:
            return cls(value_type=Map.from_json_dict(**kwargs["values"]))
        if kwargs["values"]["type"] == ANY_TYPE:
            return cls(value_type=AnyType())
        return cls(value_type=kwargs["values"]["type"])

    def _merge(self, other: BaseType) -> Map:
        if isinstance(other, AnyType) or self == other:
            return deepcopy(self)
        if not isinstance(other, Map):
            raise MlflowException(f"Can't merge map with non-map type: {type(other).__name__}")
        if isinstance(self.value_type, DataType):
            if self.value_type == other.value_type:
                return Map(value_type=self.value_type)
            raise MlflowException(
                f"Map types are incompatible for {self} with value_type={self.value_type} and "
                f"{other} with value_type={other.value_type}"
            )

        if isinstance(self.value_type, (Array, Object, Map, AnyType)):
            return Map(value_type=self.value_type._merge(other.value_type))

        raise MlflowException(f"Map type {self!r} and {other!r} are incompatible")


@experimental
class AnyType(BaseType):
    def __init__(self):
        """
        AnyType can store any json-serializable data including None values.
        For example:

        .. code-block::python

            from mlflow.types.schema import AnyType, Schema, ColSpec

            schema = Schema([ColSpec(type=AnyType(), name="id")])

        .. Note::
            AnyType should be used when the field is None, the type is not known
            at the time of data creation, or the field can have multiple types.
            e.g. for GenAI flavors, the model output could contain `None` values,
            and `AnyType` can be used to represent them.
            AnyType has no data validation at all, please be aware of this when
            using it.
        """

    def __repr__(self) -> str:
        return "Any"

    def __eq__(self, other) -> bool:
        return isinstance(other, AnyType)

    def to_dict(self):
        return {"type": ANY_TYPE}

    def _merge(self, other: BaseType) -> BaseType:
        if self == other:
            return deepcopy(self)
        if isinstance(other, DataType):
            return other
        if not isinstance(other, BaseType):
            raise MlflowException(
                f"Can't merge AnyType with {type(other).__name__}, "
                "it must be a BaseType or DataType"
            )
        # Merging AnyType with another type makes the other type optional
        return other._merge(self)


class ColSpec:
    """
    Specification of name and type of a single column in a dataset.
    """

    def __init__(
        self,
        type: ALLOWED_DTYPES,
        name: Optional[str] = None,
        required: bool = True,
    ):
        self._name = name

        self._required = required
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException(
                f"Unsupported type '{type}', expected instance of DataType or "
                f"one of {[t.name for t in DataType]}"
            )
        if not isinstance(self.type, (DataType, Array, Object, Map, AnyType)):
            raise TypeError(EXPECTED_TYPE_MESSAGE.format(arg_name="type", passed_type=self.type))

    @property
    def type(self) -> Union[DataType, Array, Object, Map, AnyType]:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    @name.setter
    def name(self, value: bool) -> None:
        self._name = value

    @experimental
    @property
    def required(self) -> bool:
        """Whether this column is required."""
        return self._required

    def to_dict(self) -> dict[str, Any]:
        d = {"type": self.type.name} if isinstance(self.type, DataType) else self.type.to_dict()
        if self.name is not None:
            d["name"] = self.name
        d["required"] = self.required
        return d

    def __eq__(self, other) -> bool:
        if isinstance(other, ColSpec):
            names_eq = (self.name is None and other.name is None) or self.name == other.name
            return names_eq and self.type == other.type and self.required == other.required
        return False

    def __repr__(self) -> str:
        required = "required" if self.required else "optional"
        if self.name is None:
            return f"{self.type!r} ({required})"
        return f"{self.name!r}: {self.type!r} ({required})"

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and
        optional `name` and `required` keys.
        """
        if not {"type"} <= set(kwargs.keys()):
            raise MlflowException("Missing keys in ColSpec JSON. Expected to find key `type`")
        if kwargs["type"] not in [ARRAY_TYPE, OBJECT_TYPE, MAP_TYPE, SPARKML_VECTOR_TYPE, ANY_TYPE]:
            return cls(**kwargs)
        name = kwargs.pop("name", None)
        required = kwargs.pop("required", None)
        if kwargs["type"] == ARRAY_TYPE:
            return cls(name=name, type=Array.from_json_dict(**kwargs), required=required)
        if kwargs["type"] == OBJECT_TYPE:
            return cls(
                name=name,
                type=Object.from_json_dict(**kwargs),
                required=required,
            )
        if kwargs["type"] == MAP_TYPE:
            return cls(name=name, type=Map.from_json_dict(**kwargs), required=required)
        if kwargs["type"] == SPARKML_VECTOR_TYPE:
            return cls(name=name, type=SparkMLVector(), required=required)
        if kwargs["type"] == ANY_TYPE:
            return cls(name=name, type=AnyType(), required=required)


class TensorInfo:
    """
    Representation of the shape and type of a Tensor.
    """

    def __init__(self, dtype: np.dtype, shape: Union[tuple, list]):
        if not isinstance(dtype, np.dtype):
            raise TypeError(
                f"Expected `dtype` to be instance of `{np.dtype}`, received `{dtype.__class__}`"
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

    def to_dict(self) -> dict[str, Any]:
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
        type: np.dtype,
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
    def required(self) -> bool:
        """Whether this tensor is required."""
        return True

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(self, inputs: list[Union[ColSpec, TensorSpec]]):
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
                f"Please choose one of {ColSpec.__name__} or {TensorSpec.__name__}"
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
        if all(x.name is None for x in inputs) and any(x.required is False for x in inputs):
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
    def inputs(self) -> list[Union[ColSpec, TensorSpec]]:
        """Representation of a dataset that defines this schema."""
        return self._inputs

    def is_tensor_spec(self) -> bool:
        """Return true iff this schema is specified using TensorSpec"""
        return self.inputs and isinstance(self.inputs[0], TensorSpec)

    def input_names(self) -> list[Union[str, int]]:
        """Get list of data names or range of indices if the schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs)]

    def required_input_names(self) -> list[Union[str, int]]:
        """Get list of required data names or range of indices if schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs) if x.required]

    @experimental
    def optional_input_names(self) -> list[Union[str, int]]:
        """Get list of optional data names or range of indices if schema has no names."""
        return [x.name or i for i, x in enumerate(self.inputs) if not x.required]

    def has_input_names(self) -> bool:
        """Return true iff this schema declares names, false otherwise."""
        return self.inputs and self.inputs[0].name is not None

    def input_types(self) -> list[Union[DataType, np.dtype, Array, Object]]:
        """Get types for each column in the schema."""
        return [x.type for x in self.inputs]

    def input_types_dict(self) -> dict[str, Union[DataType, np.dtype, Array, Object]]:
        """Maps column names to types, iff this schema declares names."""
        if not self.has_input_names():
            raise MlflowException("Cannot get input types as a dict for schema without names.")
        return {x.name: x.type for x in self.inputs}

    def input_dict(self) -> dict[str, Union[ColSpec, TensorSpec]]:
        """Maps column names to inputs, iff this schema declares names."""
        if not self.has_input_names():
            raise MlflowException("Cannot get input dict for schema without names.")
        return {x.name: x for x in self.inputs}

    def numpy_types(self) -> list[np.dtype]:
        """Convenience shortcut to get the datatypes as numpy types."""
        if self.is_tensor_spec():
            return [x.type for x in self.inputs]
        if all(isinstance(x.type, DataType) for x in self.inputs):
            return [x.type.to_numpy() for x in self.inputs]
        raise MlflowException(
            "Failed to get numpy types as some of the inputs types are not DataType."
        )

    def pandas_types(self) -> list[np.dtype]:
        """Convenience shortcut to get the datatypes as pandas types. Unsupported by TensorSpec."""
        if self.is_tensor_spec():
            raise MlflowException("TensorSpec only supports numpy types, use numpy_types() instead")
        if all(isinstance(x.type, DataType) for x in self.inputs):
            return [x.type.to_pandas() for x in self.inputs]
        raise MlflowException(
            "Failed to get pandas types as some of the inputs types are not DataType."
        )

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
                StructField(
                    name=col.name or str(i), dataType=col.type.to_spark(), nullable=not col.required
                )
                for i, col in enumerate(self.inputs)
            ]
        )

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps([x.to_dict() for x in self.inputs])

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.inputs]

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from a json string."""

        def read_input(x: dict):
            return (
                TensorSpec.from_json_dict(**x)
                if x["type"] == "tensor"
                else ColSpec.from_json_dict(**x)
            )

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
        dtype: Union[DataType, Object, str],
        default: Any,
        shape: Optional[tuple[int, ...]] = None,
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
        if not isinstance(self.dtype, (DataType, Object)):
            raise TypeError(f"'dtype' must be DataType, Object or str, got {self.dtype}")
        if self.dtype == DataType.binary:
            raise MlflowException.invalid_parameter_value(
                f"Binary type is not supported for parameters, ParamSpec '{self.name}'"
                "has dtype 'binary'",
            )

        # This line makes sure repr(self) works fine
        self._default = default
        self._default = self.validate_type_and_shape(repr(self), default, self.dtype, self.shape)

    @classmethod
    def validate_param_spec(cls, value: Any, param_spec: "ParamSpec"):
        return cls.validate_type_and_shape(
            repr(param_spec), value, param_spec.dtype, param_spec.shape
        )

    @classmethod
    def validate_type_and_shape(
        cls,
        spec: str,
        value: Any,
        value_type: Union[DataType, Object],
        shape: Optional[tuple[int, ...]],
    ):
        """
        Validate that the value has the expected type and shape.
        """
        from mlflow.models.utils import _enforce_object, _enforce_param_datatype

        def _is_1d_array(value):
            return isinstance(value, (list, np.ndarray)) and np.array(value).ndim == 1

        if shape == (-1,) and not _is_1d_array(value):
            raise MlflowException.invalid_parameter_value(
                f"Value must be a 1D array with shape (-1,) for param {spec}, "
                f"received {type(value).__name__} with ndim {np.array(value).ndim}",
            )

        try:
            if shape is None:
                if isinstance(value_type, DataType):
                    return _enforce_param_datatype(value, value_type)
                elif isinstance(value_type, Object):
                    # deepcopy to make sure the value is not mutated
                    # use _enforce_object to validate that the value matches the object schema.
                    # return the original value to preserve its type, as validation may cast it
                    # to a numpy type, but models require the original parameter type.
                    # TODO: we will drop data conversion for params in the future, including
                    # the current allowed conversions in _enforce_param_datatype
                    _enforce_object(deepcopy(value), value_type)
                    return value
            elif shape == (-1,):
                return [_enforce_param_datatype(v, value_type) for v in value]
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to validate type and shape for {spec}, error: {e}"
            )

        raise MlflowException.invalid_parameter_value(
            "Shape must be None for scalar or dictionary value, or (-1,) for 1D array value "
            f"for ParamSpec {spec}), received {shape}",
        )

    @property
    def name(self) -> str:
        """The name of the parameter."""
        return self._name

    @property
    def dtype(self) -> Union[DataType, Object]:
        """The parameter data type."""
        return self._dtype

    @property
    def default(self) -> Any:
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
        default: Union[DataType, list[DataType], None]
        shape: Optional[tuple[int, ...]]

    def to_dict(self) -> ParamSpecTypedDict:
        if self.shape is None:
            if isinstance(self.dtype, DataType) and self.dtype.name == "datetime":
                default_value = self.default.isoformat()
            else:
                default_value = self.default
        elif self.shape == (-1,):
            default_value = (
                [v.isoformat() for v in self.default]
                if self.dtype.name == "datetime"
                else self.default
            )
        result = {
            "name": self.name,
            "default": default_value,
            "shape": self.shape,
        }
        if isinstance(self.dtype, DataType):
            type_dict = {"type": self.dtype.name}
        elif isinstance(self.dtype, Object):
            type_dict = self.dtype.to_dict()
        result.update(type_dict)
        return result

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
        dtype = Object.from_json_dict(**kwargs) if dtype == OBJECT_TYPE else DataType[dtype]
        return cls(
            name=str(kwargs["name"]),
            dtype=dtype,
            default=kwargs["default"],
            shape=kwargs.get("shape"),
        )


@experimental
class ParamSchema:
    """
    Specification of parameters applicable to the model.
    ParamSchema is represented as a list of :py:class:`ParamSpec`.
    """

    def __init__(self, params: list[ParamSpec]):
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
    def _find_duplicates(params: list[ParamSpec]) -> list[str]:
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
    def params(self) -> list[ParamSpec]:
        """Representation of ParamSchema as a list of ParamSpec."""
        return self._params

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from a json string."""
        return cls([ParamSpec.from_json_dict(**x) for x in json.loads(json_str)])

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.params]

    def __eq__(self, other) -> bool:
        if isinstance(other, ParamSchema):
            return self.params == other.params
        return False

    def __repr__(self) -> str:
        return repr(self.params)


def _map_field_type(field):
    field_type_mapping = {
        bool: "boolean",
        int: "long",  # int is mapped to long to support 64-bit integers
        builtins.float: "float",
        str: "string",
        bytes: "binary",
        dt.date: "datetime",
    }
    return field_type_mapping.get(field)


def _get_dataclass_annotations(cls) -> dict[str, Any]:
    """
    Given a dataclass or an instance of one, collect annotations from it and all its parent
    dataclasses.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass.")

    annotations = {}
    effective_class = cls if isinstance(cls, type) else type(cls)

    # Reverse MRO so subclass overrides are captured last
    for base in reversed(effective_class.__mro__):
        # Only capture supers that are dataclasses
        if is_dataclass(base) and hasattr(base, "__annotations__"):
            annotations.update(base.__annotations__)
    return annotations


@experimental
def convert_dataclass_to_schema(dataclass):
    """
    Converts a given dataclass into a Schema object. The dataclass must include type hints
    for all its fields. Fields can be of basic types, other dataclasses, or Lists/Optional of
    these types. Union types are not supported. Only the top-level fields are directly converted
    to ColSpecs, while nested fields are converted into nested Object types.
    """

    inputs = []

    for field_name, field_type in _get_dataclass_annotations(dataclass).items():
        # Determine the type and handle Optional and List correctly
        is_optional = False
        effective_type = field_type

        if get_origin(field_type) == Union:
            if type(None) in get_args(field_type) and len(get_args(field_type)) == 2:
                # This is an Optional type; determine the effective type excluding None
                is_optional = True
                effective_type = next(t for t in get_args(field_type) if t is not type(None))
            else:
                raise MlflowException(
                    "Only Optional[...] is supported as a Union type in dataclass fields"
                )

        if get_origin(effective_type) == list:
            # It's a list, check the type within the list
            list_type = get_args(effective_type)[0]
            if is_dataclass(list_type):
                dtype = _convert_dataclass_to_nested_object(list_type)  # Convert to nested Object
                inputs.append(
                    ColSpec(type=Array(dtype=dtype), name=field_name, required=not is_optional)
                )
            else:
                if dtype := _map_field_type(list_type):
                    inputs.append(
                        ColSpec(
                            type=Array(dtype=dtype),
                            name=field_name,
                            required=not is_optional,
                        )
                    )
                else:
                    raise MlflowException(
                        f"List field type {list_type} is not supported in dataclass"
                        f" {dataclass.__name__}"
                    )
        elif is_dataclass(effective_type):
            # It's a nested dataclass
            dtype = _convert_dataclass_to_nested_object(effective_type)  # Convert to nested Object
            inputs.append(
                ColSpec(
                    type=dtype,
                    name=field_name,
                    required=not is_optional,
                )
            )
        # confirm the effective type is a basic type
        elif dtype := _map_field_type(effective_type):
            # It's a basic type
            inputs.append(
                ColSpec(
                    type=dtype,
                    name=field_name,
                    required=not is_optional,
                )
            )
        else:
            raise MlflowException(
                f"Unsupported field type {effective_type} in dataclass {dataclass.__name__}"
            )

    return Schema(inputs=inputs)


def _convert_dataclass_to_nested_object(dataclass):
    """
    Convert a nested dataclass to an Object type used within a ColSpec.
    """
    properties = []
    for field_name, field_type in dataclass.__annotations__.items():
        properties.append(_convert_field_to_property(field_name, field_type))
    return Object(properties=properties)


def _convert_field_to_property(field_name, field_type):
    """
    Helper function to convert a single field to a Property object suitable for inclusion in an
    Object.
    """

    is_optional = False
    effective_type = field_type

    if get_origin(field_type) == Union and type(None) in get_args(field_type):
        is_optional = True
        effective_type = next(t for t in get_args(field_type) if t is not type(None))

    if get_origin(effective_type) == list:
        list_type = get_args(effective_type)[0]
        return Property(
            name=field_name,
            dtype=Array(dtype=_map_field_type(list_type)),
            required=not is_optional,
        )
    elif is_dataclass(effective_type):
        return Property(
            name=field_name,
            dtype=_convert_dataclass_to_nested_object(effective_type),
            required=not is_optional,
        )
    else:
        return Property(
            name=field_name,
            dtype=_map_field_type(effective_type),
            required=not is_optional,
        )
