from enum import Enum

from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema
from mlflow.types.type_hints import _infer_schema_from_list_type_hint


class Color(Enum):
    RED = "red"
    BLUE = "blue"


def test_infer_schema_from_enum_type_hint():
    schema = _infer_schema_from_list_type_hint(list[Color])
    assert schema == Schema([ColSpec(type=DataType.string, enum=["red", "blue"])])