from enum import Enum

import json
import numpy as np
import os
import pandas as pd

from mlflow.pyfunc.utils import get_jsonable_obj


class ModelSignature(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self):
        return {
            "inputs": json.dumps([col.to_dict() for col in self.inputs]),
            "outputs": json.dumps([col.to_dict() for col in self.outputs])
        }

    @classmethod
    def from_dict(cls, signature_dict):
        def schema_from_json(json_string):
            from mlflow.models.signature import ColSpec
            return [ColSpec(**x) for x in json.loads(json_string)]

        inputs = schema_from_json(signature_dict["inputs"])
        outputs = schema_from_json(signature_dict["inputs"])
        return cls(inputs, outputs)

    def __str__(self):
        import yaml
        return yaml.safe_dump(self.to_dict())


class DataType(Enum):
    boolean = 1
    integer = 2
    long = 3
    float = 4
    double = 5
    string = 6
    binary = 7

    def __str__(self):
        return self.name


class ColSpec(object):
    def __init__(self, name: str, type: DataType):
        self.name = name
        self.type = getattr(DataType, type) if isinstance(type, str) else type

    def to_dict(self):
        return {"name": self.name, "type": self.type.name}

    def __str__(self):
        return "{name}: {type}".format(name=self.name, type=self.type)


def _map_numpy_dtype(col):
    if not isinstance(col, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(col)))
    if len(col.shape) > 1:
        raise Exception("Multidimensional arrays (aka tensors) are not supported.")
    if col.dtype.kind == "b":
        return DataType.boolean
    elif col.dtype.kind == "i" or col.dtype.kind == "u":
        if col.dtype.itemsize < 4 or col.dtype.kind == "i" and col.dtype.itemsize == 4:
            return DataType.integer
        elif col.dtype.itemsize < 8 or col.dtype.kind == "i" and col.dtype.itemsize == 8:
            return DataType.long
    elif col.dtype.kind == "f":
        if col.dtype.itemsize <= 4:
            return DataType.float
        elif col.dtype.itemsize <= 8:
            return DataType.double
    elif col.dtype.kind == "U":
        return DataType.string
    elif col.dtype.kind == "S":
        return DataType.binary
    elif col.dtype.kind == "O":
        first_elem = col[0]
        if isinstance(first_elem, bytearray) and all(
                [isinstance(x, bytearray) for x in col]):
            return DataType.binary
        elif isinstance(first_elem, str) and all([isinstance(x, str) for x in col]):
            return DataType.string
    raise Exception("Unsupported type", col.dtype)


def _map_spark_type(x):
    import pyspark.sql.types
    if isinstance(x, pyspark.sql.types.NumericType):
        if isinstance(x, pyspark.sql.types.IntegralType):
            if isinstance(x, pyspark.sql.types.LongType):
                return DataType.long
            else:
                return DataType.integer
        elif isinstance(x, pyspark.sql.types.FloatType):
            return DataType.float
        elif isinstance(x, pyspark.sql.types.DoubleType):
            return DataType.double

    elif isinstance(x, pyspark.sql.types.StringType):
        return DataType.string
    elif isinstance(x, pyspark.sql.types.BinaryType):
        return DataType.binary
    else:
        raise Exception("Unsupported Spark Type '{}', MLflow schema is only supported for scalar "
                        "Spark types.")


def _is_spark_df(x):
    try:
        import pyspark.sql.DataFrame
        return isinstance(x, pyspark.sql.DataFrame)
    except ModuleNotFoundError:
        return False


def _infer_schema(data):
    if isinstance(data, pd.DataFrame):
        return [ColSpec(col, _map_numpy_dtype(data[col].values)) for col in data.columns]
    elif isinstance(data, np.ndarray):
        array_type = _map_numpy_dtype(data)
        if len(data.shape) == 1:
            return [ColSpec(None, array_type)]
        if len(data.shape) == 2:
            return [ColSpec(None, array_type) for _ in range(data.shape[1])]
        if len(data.shape) > 2:
            raise Exception("Multidimensional arrays (aka tensors) are not supported.")
    elif isinstance(data, dict):
        return [ColSpec(col_name, _map_numpy_dtype(col_values))
                for col_name, col_values in data.items()]
    elif _is_spark_df(data):
        return [ColSpec(field.name, _map_spark_type(field.type)) for field in data.schema.fields]


def save_example(path, data):
    example_file = "input_example.json"
    if isinstance(data, pd.DataFrame):
        data.to_json(os.path.join(path, example_file), orient="records")
        return example_file
    elif isinstance(data, dict):

        jsonable_data = {key: get_jsonable_obj(value) for key, value in data.items()}
        json.dump(jsonable_data, os.path.join(path, example_file))
        return example_file
    elif isinstance(data, np.ndarray):
        json.dump(data.tolist(), os.path.join(path, example_file))
        return example_file
    else:
        raise TypeError("Unsupported example type, expected one of "
                        "(pandas.DataFrame, dictionary, numpy.ndarray), got '{}'"
                        .format(type(data)))


def infer_signature(train, model_prediction):
    return ModelSignature(_infer_schema(train), _infer_schema(model_prediction))
