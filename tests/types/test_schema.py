import datetime
import json
import re

import numpy as np
import pytest

from mlflow.exceptions import MlflowException
from mlflow.types import DataType
from mlflow.types.schema import (
    Array,
    ColSpec,
    Object,
    ParamSchema,
    ParamSpec,
    Property,
    Schema,
    TensorSpec,
)


def test_datatype_type_check():
    assert DataType.is_string("string")

    assert DataType.is_integer(1)
    assert DataType.is_integer(np.int32(1))
    assert not DataType.is_integer(np.int64(1))
    # Note that isinstance(True, int) returns True
    assert not DataType.is_integer(True)

    assert DataType.is_long(1)
    assert DataType.is_long(np.int64(1))
    assert not DataType.is_long(np.int32(1))

    assert DataType.is_boolean(True)
    assert DataType.is_boolean(np.bool_(True))
    assert not DataType.is_boolean(1)

    assert DataType.is_double(1.0)
    assert DataType.is_double(np.float64(1.0))
    assert not DataType.is_double(np.float32(1.0))

    assert DataType.is_float(1.0)
    assert DataType.is_float(np.float32(1.0))
    assert not DataType.is_float(np.float64(1.0))

    assert DataType.is_datetime(datetime.date(2023, 6, 26))
    assert DataType.is_datetime(np.datetime64("2023-06-26 00:00:00"))
    assert not DataType.is_datetime("2023-06-26 00:00:00")


def test_col_spec():
    a1 = ColSpec("string", "a")
    a2 = ColSpec(DataType.string, "a")
    a3 = ColSpec(DataType.integer, "a")
    assert a1 != a3
    b1 = ColSpec(DataType.string, "b")
    assert b1 != a1
    assert a1 == a2
    with pytest.raises(MlflowException, match="Unsupported type 'unsupported'"):
        ColSpec("unsupported")
    a4 = ColSpec(**a1.to_dict())
    assert a4 == a1
    assert ColSpec(**json.loads(json.dumps(a1.to_dict()))) == a1
    a5 = ColSpec("string")
    a6 = ColSpec("string", None)
    assert a5 == a6
    assert ColSpec(**json.loads(json.dumps(a5.to_dict()))) == a5


def test_tensor_spec():
    a1 = TensorSpec(np.dtype("float64"), (-1, 3, 3), "a")
    a2 = TensorSpec(np.dtype("float"), (-1, 3, 3), "a")  # float defaults to float64
    a3 = TensorSpec(np.dtype("float"), [-1, 3, 3], "a")
    a4 = TensorSpec(np.dtype("int"), (-1, 3, 3), "a")
    assert a1 == a2
    assert a1 == a3
    assert a1 != a4
    b1 = TensorSpec(np.dtype("float64"), (-1, 3, 3), "b")
    assert b1 != a1
    with pytest.raises(TypeError, match="Expected `type` to be instance"):
        TensorSpec("Unsupported", (-1, 3, 3), "a")
    with pytest.raises(TypeError, match="Expected `shape` to be instance"):
        TensorSpec(np.dtype("float64"), np.array([-1, 2, 3]), "b")
    with pytest.raises(
        MlflowException,
        match="MLflow does not support size information in flexible numpy data types",
    ):
        TensorSpec(np.dtype("<U10"), (-1,), "b")

    a5 = TensorSpec.from_json_dict(**a1.to_dict())
    assert a5 == a1
    assert TensorSpec.from_json_dict(**json.loads(json.dumps(a1.to_dict()))) == a1
    a6 = TensorSpec(np.dtype("float64"), (-1, 3, 3))
    a7 = TensorSpec(np.dtype("float64"), (-1, 3, 3), None)
    assert a6 == a7
    assert TensorSpec.from_json_dict(**json.loads(json.dumps(a6.to_dict()))) == a6


def test_schema_creation():
    # can create schema with named col specs
    Schema([ColSpec("double", "a"), ColSpec("integer", "b")])

    # can create schema with unnamed col specs
    Schema([ColSpec("double"), ColSpec("integer")])

    # can create schema with multiple named tensor specs
    Schema([TensorSpec(np.dtype("float64"), (-1,), "a"), TensorSpec(np.dtype("uint8"), (-1,), "b")])

    # can create schema with single unnamed tensor spec
    Schema([TensorSpec(np.dtype("float64"), (-1,))])

    # combination of tensor and col spec is not allowed
    with pytest.raises(MlflowException, match="Please choose one of"):
        Schema([TensorSpec(np.dtype("float64"), (-1,)), ColSpec("double")])

    # combination of named and unnamed inputs is not allowed
    with pytest.raises(
        MlflowException, match="Creating Schema with a combination of named and unnamed inputs"
    ):
        Schema(
            [TensorSpec(np.dtype("float64"), (-1,), "blah"), TensorSpec(np.dtype("float64"), (-1,))]
        )

    with pytest.raises(
        MlflowException, match="Creating Schema with a combination of named and unnamed inputs"
    ):
        Schema([ColSpec("double", "blah"), ColSpec("double")])

    # multiple unnamed tensor specs is not allowed
    with pytest.raises(
        MlflowException, match="Creating Schema with multiple unnamed TensorSpecs is not supported"
    ):
        Schema([TensorSpec(np.dtype("double"), (-1,)), TensorSpec(np.dtype("double"), (-1,))])


def test_schema_creation_errors():
    with pytest.raises(MlflowException, match=r"Creating Schema with empty inputs is not allowed."):
        Schema([])

    with pytest.raises(MlflowException, match=r"Inputs of Schema must be a list, got type dict"):
        Schema({"col1": ColSpec(DataType.string)})


def test_param_schema_find_duplicates():
    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param1']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
            ]
        )

    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param1']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
                ParamSpec("param1", DataType.string, "default1", None),
            ]
        )

    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param3']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
                ParamSpec("param3", DataType.string, "default3", None),
                ParamSpec("param3", DataType.string, "default3", None),
            ]
        )


def test_param_spec_to_and_from_dict():
    spec = ParamSpec("str_param", DataType.string, "str_a", None)
    assert spec.to_dict() == {
        "name": "str_param",
        "type": "string",
        "default": "str_a",
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("str_array", DataType.string, ["str_a", "str_b"], (-1,))
    assert spec.to_dict() == {
        "name": "str_array",
        "type": "string",
        "default": ["str_a", "str_b"],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("int_param", DataType.integer, np.int32(1), None)
    assert spec.to_dict() == {
        "name": "int_param",
        "type": "integer",
        "default": 1,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("int_array", DataType.integer, [np.int32(1), np.int32(2)], (-1,))
    assert spec.to_dict() == {
        "name": "int_array",
        "type": "integer",
        "default": [1, 2],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("bool_param", DataType.boolean, True, None)
    assert spec.to_dict() == {
        "name": "bool_param",
        "type": "boolean",
        "default": True,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("bool_array", DataType.boolean, [True, False], (-1,))
    assert spec.to_dict() == {
        "name": "bool_array",
        "type": "boolean",
        "default": [True, False],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("double_param", DataType.double, 1.0, None)
    assert spec.to_dict() == {
        "name": "double_param",
        "type": "double",
        "default": 1.0,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("double_array", DataType.double, [1.0, 2.0], (-1,))
    assert spec.to_dict() == {
        "name": "double_array",
        "type": "double",
        "default": [1.0, 2.0],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("float_param", DataType.float, np.float32(0.1), None)
    assert spec.to_dict() == {
        "name": "float_param",
        "type": "float",
        "default": float(np.float32(0.1)),
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("float_array", DataType.float, [np.float32(0.1), np.float32(0.2)], (-1,))
    assert spec.to_dict() == {
        "name": "float_array",
        "type": "float",
        "default": [float(np.float32(0.1)), float(np.float32(0.2))],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("long_param", DataType.long, 100, None)
    assert spec.to_dict() == {
        "name": "long_param",
        "type": "long",
        "default": 100,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("long_array", DataType.long, [100, 200], (-1,))
    assert spec.to_dict() == {
        "name": "long_array",
        "type": "long",
        "default": [100, 200],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec(
        "datetime_param", DataType.datetime, np.datetime64("2023-06-26 00:00:00"), None
    )
    assert spec.to_dict() == {
        "name": "datetime_param",
        "type": "datetime",
        "default": "2023-06-26T00:00:00",
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec(
        "datetime_array",
        DataType.datetime,
        [np.datetime64("2023-06-26 00:00:00"), np.datetime64("2023-06-27 00:00:00")],
        (-1,),
    )
    assert spec.to_dict() == {
        "name": "datetime_array",
        "type": "datetime",
        "default": ["2023-06-26T00:00:00", "2023-06-27T00:00:00"],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec


def test_param_spec_from_dict_backward_compatibility():
    spec = ParamSpec("str_param", DataType.string, "str_a", None)
    spec_json = json.dumps(
        {
            "name": "str_param",
            "dtype": "string",
            "default": "str_a",
            "shape": None,
        }
    )
    assert ParamSpec.from_json_dict(**json.loads(spec_json)) == spec


def test_object_construction_with_errors():
    with pytest.raises(MlflowException, match=r"Expected properties to be a list, got type dict"):
        Object({"p1": Property("p1", DataType.string)})

    with pytest.raises(
        MlflowException, match=r"Creating Object with empty properties is not allowed."
    ):
        Object([])

    properties = [
        Property("p1", DataType.string),
        Property("p2", DataType.binary),
        {"invalid_type": "value"},
    ]
    with pytest.raises(MlflowException, match=r"Expected values to be instance of Property"):
        Object(properties)

    properties = [
        Property("p1", DataType.string),
        Property("p2", DataType.binary),
        Property("p2", DataType.boolean),
    ]
    with pytest.raises(MlflowException, match=r"Found duplicated property names: {'p2'}"):
        Object(properties)


def test_object_to_and_from_dict():
    properties = []
    dict_prop = {}
    for data_type in DataType:
        properties.append(Property(f"name_{data_type.name}", data_type))
        dict_prop[f"name_{data_type.name}"] = {"type": data_type.name, "required": True}
    obj = Object(properties)
    assert obj.to_dict() == {
        "type": "object",
        "properties": dict(sorted(dict_prop.items())),
    }
    assert Object.from_json_dict(**json.loads(json.dumps(obj.to_dict()))) == obj


def test_object_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Object JSON. Expected to find keys `properties` and `type`",
    ):
        Object.from_json_dict(**{"type": "object"})

    with pytest.raises(
        MlflowException, match=r"Type mismatch, Object expects `object` as the type"
    ):
        Object.from_json_dict(**{"type": "array", "properties": {}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Object.from_json_dict(**{"type": "object", "properties": "invalid_type"})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Object.from_json_dict(
            **{
                "type": "object",
                "properties": {"p1": {"type": "string"}, "p2": "invalid_type"},
            }
        )


def test_object_merge():
    obj1 = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double),
        ]
    )
    obj1_dict = obj1.to_dict()
    obj2 = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="c", dtype=DataType.boolean),
        ]
    )
    obj2_dict = obj2.to_dict()
    updated_obj = obj1._merge(obj2)
    assert updated_obj == Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double, required=False),
            Property(name="c", dtype=DataType.boolean, required=False),
        ]
    )
    assert obj1.to_dict() == obj1_dict
    assert obj2.to_dict() == obj2_dict


def test_repr_of_objects():
    obj = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double, required=False),
            Property(name="c", dtype=Array(DataType.long)),
            Property(name="d", dtype=Object([Property("d1", DataType.string)])),
        ]
    )
    obj_repr = (
        "{a: string (required), b: double (optional), c: Array(long) "
        "(required), d: {d1: string (required)} (required)}"
    )
    assert repr(obj) == obj_repr

    arr = Array(obj)
    assert repr(arr) == f"Array({obj_repr})"


@pytest.mark.parametrize("data_type", DataType)
def test_property_to_and_from_dict(data_type):
    prop = Property("data", data_type, True)
    assert prop.to_dict() == {"data": {"type": data_type.name, "required": True}}
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop

    # test array
    prop = Property("arr", Array(data_type), False)
    assert prop.to_dict() == {
        "arr": {
            "type": "array",
            "items": {"type": data_type.name},
            "required": False,
        },
    }
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop

    # test object
    prop = Property("data", Object([Property("p", data_type)]))
    assert prop.to_dict() == {
        "data": {
            "type": "object",
            "properties": {"p": {"type": data_type.name, "required": True}},
            "required": True,
        },
    }
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop


def test_property_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Expected Property JSON to contain a single key as name, got 2 keys.",
    ):
        Property.from_json_dict(**{"p1": {}, "p2": {}})

    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Property `p`. Expected to find key `type`",
    ):
        Property.from_json_dict(**{"p": {}})

    with pytest.raises(
        MlflowException,
        match=r"Unsupported type 'invalid_type', expected instance of DataType, Array, Object or ",
    ):
        Property.from_json_dict(**{"p": {"type": "invalid_type"}})

    # test array
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Array JSON. Expected to find keys `items` and `type`",
    ):
        Property.from_json_dict(**{"p": {"type": "array"}})

    with pytest.raises(
        MlflowException,
        match=r"Unsupported type 'invalid_type', expected instance of DataType, Array, Object or ",
    ):
        Property.from_json_dict(**{"p": {"type": "array", "items": {"type": "invalid_type"}}})

    with pytest.raises(
        MlflowException,
        match=r"Expected items to be a dictionary of Object JSON",
    ):
        Property.from_json_dict(**{"p": {"type": "array", "items": "invalid_items_type"}})

    # test object
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Object JSON. Expected to find keys `properties` and `type`",
    ):
        Property.from_json_dict(**{"p": {"type": "object"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Property.from_json_dict(**{"p": {"type": "object", "properties": "invalid_type"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Property.from_json_dict(
            **{
                "p": {
                    "type": "object",
                    "properties": {"p1": {"type": "string"}, "p2": "invalid_type"},
                }
            }
        )


def test_property_merge():
    prop1 = Property(
        name="a",
        dtype=Object(properties=[Property(name="a", dtype=DataType.string, required=False)]),
    )
    prop1_dict = prop1.to_dict()
    prop2 = Property(
        name="a",
        dtype=Object(
            properties=[
                Property(name="a", dtype=DataType.string),
                Property(name="b", dtype=DataType.double),
            ]
        ),
    )
    prop2_dict = prop2.to_dict()
    updated_prop = prop1._merge(prop2)
    assert updated_prop == Property(
        name="a",
        dtype=Object(
            properties=[
                Property(name="a", dtype=DataType.string, required=False),
                Property(name="b", dtype=DataType.double, required=False),
            ]
        ),
    )
    assert prop1.to_dict() == prop1_dict
    assert prop2.to_dict() == prop2_dict


@pytest.mark.parametrize("data_type", DataType)
def test_array_to_and_from_dict(data_type):
    arr = Array(data_type)
    assert arr.to_dict() == {"type": "array", "items": {"type": data_type.name}}
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr

    # test object
    arr = Array(Object([Property("p", data_type)]))
    assert arr.to_dict() == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"p": {"type": data_type.name, "required": True}},
        },
    }
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr


def test_array_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Array JSON. Expected to find keys `items` and `type`",
    ):
        Array.from_json_dict(**{"type": "array"})

    with pytest.raises(MlflowException, match=r"Type mismatch, Array expects `array` as the type"):
        Array.from_json_dict(**{"type": "object", "items": "string"})

    with pytest.raises(MlflowException, match=r"Expected items to be a dictionary of Object JSON"):
        Array.from_json_dict(**{"type": "array", "items": "string"})

    with pytest.raises(
        MlflowException,
        match=r"Unsupported type 'invalid_type', expected instance of DataType, Array, Object or ",
    ):
        Array.from_json_dict(**{"type": "array", "items": {"type": "invalid_type"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Array.from_json_dict(**{"type": "array", "items": {"type": "object", "properties": []}})


def test_nested_array_object_to_and_from_dict():
    arr = Array(
        Object(
            [
                Property("p", DataType.string),
                Property(
                    "arr",
                    Array(
                        Object(
                            [
                                Property("p2", DataType.boolean, required=False),
                                Property("arr2", Array(DataType.long)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )
    assert arr.to_dict() == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "p": {"type": "string", "required": True},
                "arr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "p2": {"type": "boolean", "required": False},
                            "arr2": {
                                "type": "array",
                                "items": {"type": "long"},
                                "required": True,
                            },
                        },
                    },
                    "required": True,
                },
            },
        },
    }
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr
