import pandas as pd
from dataclasses import dataclass
from enum import Enum
import pytest

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _enforce_schema
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, convert_dataclass_to_schema


class Color(Enum):
    RED = "red"
    BLUE = "blue"


def test_convert_dataclass_to_schema_enum():
    @dataclass
    class Input:
        color: Color

    schema = convert_dataclass_to_schema(Input)
    assert schema == Schema([ColSpec(type=DataType.string, name="color", enum=["red", "blue"])])


def test_enforce_schema_enum():
    schema = Schema([ColSpec(type=DataType.string, name="color", enum=["red", "blue"])])
    df = pd.DataFrame({"color": ["red", "blue"]})
    _enforce_schema(df, schema)
    with pytest.raises(MlflowException, match="enum"):
        _enforce_schema(pd.DataFrame({"color": ["green"]}), schema)