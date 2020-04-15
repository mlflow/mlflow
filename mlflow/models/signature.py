"""
The :py:mod:`mlflow.models.signature` module provides an API for specification of model signature
and of model example input.

Model signature defines schema of model input and output. The schema is based on a simplified data
frame model defined by a list of (optionally) named and (mandatory) typed columns. The currently
supported data types are defined by :py:class:`DataType` and consists of basic scalar types and
string and binary string. Neither arrays (and multidimensional arrays) nor structured types
(struct, map) are supported at this time.

Input example specifies one or more valid inputs (observations) to the model. The example is stored
as json and so the examples are limited to jsonable content with the exception of binary data
provided the signature is defined. If the example includes binary data and model signature is
defined, the binary data is base64 encoded before producing output.

User can construct schema by hand or infer it from a dataset. Input examples are provided as is
(pandas.DataFrame, numpy.ndarray, dictinary or list) and are json serialized  and stored as part of
the model artifacts.
"""

import base64
import importlib
import os
from enum import Enum

import json
import numpy as np
import pandas as pd
from typing import List, Dict, TypeVar

from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import NumpyEncoder


class DataType(Enum):
    """
    Data types supported by Mlflow model signature.
    """

    def __new__(cls, value, numpy_type):
        res = object.__new__(cls)
        res._value_ = value
        res._numpy_type = numpy_type
        return res

    boolean = (1, np.bool)
    integer = (2, np.int32)
    long = (3, np.int64)
    float = (4, np.float32)
    double = (5, np.float64)
    string = (6, np.str)
    binary = (7, np.bytes_)

    def __str__(self):
        return self.name

    def to_numpy(self):
        return self._numpy_type


class ColSpec(object):
    """
    Specification of a column used in model signature.
    Declares data type and optionally a name.
    """

    def __init__(self, name: str, type: DataType):  # pylint: disable=redefined-builtin
        self.name = name
        try:
            self.type = getattr(DataType, type) if isinstance(type, str) else type
        except AttributeError:
            raise MlflowException("Unsupported type '{0}', expected instance of DataType or "
                                  "one of {1}".format(type, [t.name for t in DataType]))

    def to_dict(self) -> Dict[str, str]:
        """
        Serialize into a jsonable dictionary.
        :return: dictionary representation of the column spec.
        """
        return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        names_eq = self.name is None and other.name is None or self.name == other.name
        return names_eq and self.type == other.type

    def __repr__(self) -> str:
        return "{name}: {type}".format(name=self.name, type=self.type)


class Schema(object):
    """
    Schema specifies column types (:py:class:`DataType`) in a dataset.
    """

    def __init__(self, cols: List[ColSpec]):
        self._cols = cols

    @property
    def columns(self) -> List[ColSpec]:
        return self._cols

    def column_names(self) -> List[str]:
        return [x.name or i for i, x in enumerate(self.columns)]

    def column_types(self) -> List[DataType]:
        return [x.type for x in self._cols]

    def numpy_types(self) -> List[np.dtype]:
        return [x.type.to_numpy() for x in self.columns]

    def to_json(self) -> str:
        return json.dumps([x.to_dict() for x in self.columns])

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return self.columns == other.columns
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.columns)

    @classmethod
    def from_json(cls, json_str: str):
        return cls([ColSpec(**x) for x in json.loads(json_str)])


class ModelSignature(object):
    """
    ModelSignature specifies schema of model's inputs and outputs.

    The current supported schema for both the input and the output is a data-frame like schema
    defined as a list of column specification :py:class:`ColSpec`. Columns can be named and must
    specify their data type. Currently the list of supported types is limited to scalar data types
    defined in :py:class:`DataType` enum.

    ModelSignature can be inferred from training dataset and model predictions using
    :py:func:`mlflow.models.signature.infer_signature`, or alternatively constructed by hand by
    passing a lists of input and output column specifications.
    """

    def __init__(self, inputs: Schema, outputs: Schema = None):
        if not isinstance(inputs, Schema):
            raise TypeError("inputs must be mlflow.models.signature.Schema, got '{}'".format(
                type(inputs)))
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError("outputs must be either None or mlflow.models.signature.Schema, "
                            "got '{}'".format(type(inputs)))
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, str]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in a MLmofel yaml file.

        :return: dictionary representation with input and output shcema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, str]):
        """
        Deserialize from dictionary representation.

        :param signature_dict: Dictionary representation of model signature.
                               Expected dictionary format:
                               `{'inputs': <json string>, 'outputs': <json string>" }`

        :return: ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        if "outputs" in signature_dict and signature_dict["outputs"] is not None:
            outputs = Schema.from_json(signature_dict["outputs"])
            return cls(inputs, outputs)
        else:
            return cls(inputs)

    def __eq__(self, other) -> bool:
        return self.inputs == other.inputs and self.outputs == other.outputs

    def __repr__(self) -> str:
        return json.dumps({"ModelSignature": self.to_dict()}, indent=2)


ModelInputExample = TypeVar('ModelInputExample', pd.DataFrame, np.ndarray, dict, list)
MlflowModelDataset = TypeVar('MlflowModelDataset', pd.DataFrame, np.ndarray, Dict[str, np.ndarray])


def infer_signature(model_input: MlflowModelDataset,
                    model_output: MlflowModelDataset = None) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).
    This method captures the column names and data types from the user data. The signature
    represents model input and output as dataframes with (optionally) named columns and data type
    specified as one of types defined in :py:class:`DataType`. This method will raise
    an exception if the user data contains incompatible types or is not passed in one of the
    supported formats (containers).

    Supported input:

    container types:
      - pandas.DataFrame
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame

    element (scalar) types:
      - those that can be translated to one of MLflow's DataType.

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    :param model_input: Valid input to the model. E.g. (a subset of) training dataset.
    :param model_output: Valid model output. Model predictions for the (subset of) training dataset.
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    outputs = _infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)


def save_example(path: str, input_example: ModelInputExample, schema: Schema = None) -> str:
    """
    Save mlflow example into a file on a given path and return the resulting filename.

    MLflow examples are stored as json and so the input data must be jsonable. If the input data
    contains binary columns, the caller must provide schema and binary columns will be  base64
    encoded before generating the output.

    :param path: Path where to store the example.
    :param input_example: Data with the input example(s).
    :param schema: Input example data types.
    :return: Filename of the stored example.
    """
    if isinstance(input_example, dict):
        if all([np.isscalar(x) for x in input_example.values()]):
            input_example = pd.DataFrame({x: [v] for x, v in input_example.items()})
    elif isinstance(input_example, list):
        input_example = pd.DataFrame(input_example)
    if isinstance(input_example, pd.DataFrame):
        example_filename = "input_dataframe_example.json"
    elif isinstance(input_example, dict):
        example_filename = "input_dictionary_example.json"
    elif isinstance(input_example, np.ndarray):
        example_filename = "input_array_example.json"
    else:
        raise TypeError("Unexpected type of input_example. Expected one of "
                        "(pandas.DataFrame, numpy.ndarray, dict, list), got {}".format(
                          type(input_example)))
    with open(os.path.join(path, example_filename), "w") as f:
        to_json(input_example, pandas_orient="split", schema=schema, output_stream=f)
    return example_filename


def from_json(path_or_str, schema: Schema = None, pandas_orient: str = "records") -> pd.DataFrame:
    """
    Read data frame back from json.

    The data is always read as DataFrame even if it was written out as other supported data types (
    numpy.ndarray or {str->numpy.ndarray}. If the data was saved with a schema, caller should pass
    the same schema to ensure correct data parsing (e.g. binary columns need to be base64 decoded).

    :param path_or_str: Path to a json file or a json string.
    :param schema: Mlflow schema used when parsing the data.
    :param pandas_orient: pandas data frame convention used to store the data.
    :return: pandas.DataFrame.
    """
    if schema is not None:
        dtypes = dict(zip(schema.column_names(), schema.column_types()))
        df = pd.read_json(path_or_str, orient=pandas_orient, dtype=dtypes)[schema.column_names()]
        binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]

        def base64decode(x):
            return base64.decodebytes(x.encode("ascii"))

        for i in binary_cols:
            col = df.columns[i]
            df[col] = np.array(df[col].map(base64decode), dtype=np.bytes_)
            return df
    else:
        return pd.read_json(path_or_str, orient=pandas_orient, dtype=False)


def to_json(data: MlflowModelDataset, pandas_orient: str = "records", schema: Schema = None,
            output_stream=None):
    """Write data out as json.
    The data can not contain any non-jsonable columns except for 'binary' columns. If data contains
    binary column(s), the caller must pass schema otherwise TypeError is raised. Binary columns are
    base64 encoded.

    :param data: data to be converted, works with pandas and numpy, rest will be returned as is.
    :param pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
                          dictionary using this Pandas serialization orientation.
    :param schema: Schema of the data. It is required if the dataset contains data types that are
                   not jsonable (binary).
    :param output_stream: File-like. Output is written into the output stream if provided. It is
                          returned as a string otherwise.
    """

    if schema is not None:
        binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]
    else:
        binary_cols = []

    def base64encode(x):
        return base64.encodebytes(x).decode("ascii")

    def base64_encode_ndarray(x):
        base64encode_vec = np.vectorize(base64encode)
        if len(x.shape) == 1:
            return base64encode_vec(x)
        else:
            y = x.copy()
            for i in binary_cols:
                y[..., i] = base64encode_vec(x[..., i])
            return y

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if isinstance(data, pd.DataFrame):
        if binary_cols:
            data = data.copy()
            for i in binary_cols:
                col = data.columns[i]
                data[col] = data[col].map(base64encode)
        data = data.to_dict(orient=pandas_orient)
    elif isinstance(data, dict):
        new_data = {}
        keys = list(data.keys())
        binary_col_names = set([keys[i] for i in binary_cols])
        for col in keys:
            if not isinstance(data[col], np.ndarray):
                raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(data[col])))
            if col in binary_col_names:
                new_data[col] = base64_encode_ndarray(data[col]).tolist()
            else:
                new_data[col] = data[col].tolist()
        data = new_data
    elif isinstance(data, np.ndarray):
        if binary_cols:
            data = base64_encode_ndarray(data).tolist()
        else:
            data = data.tolist()

    if output_stream is not None:
        json.dump(data, output_stream, cls=NumpyEncoder)
    else:
        return json.dumps(data, cls=NumpyEncoder)


def _map_numpy_dtype(col: np.ndarray) -> DataType:
    if not isinstance(col, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(col)))
    if len(col.shape) > 1:
        raise MlflowException("Expected 1d array, got array with shape {}".format(col.shape))
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
        if isinstance(first_elem, bytearray) or isinstance(first_elem, bytes) and all(
                [isinstance(x, bytearray) or isinstance(x, bytes) for x in col]):
            return DataType.binary
        elif isinstance(first_elem, str) and all([isinstance(x, str) for x in col]):
            return DataType.string
        elif isinstance(first_elem, int) and all([isinstance(x, int) for x in col]):
            return DataType.long
        elif isinstance(first_elem, float) and all([isinstance(x, float) for x in col]):
            return DataType.double
        else:
            raise MlflowException("unsupported element type {} ".format(type(first_elem)))
    raise MlflowException("Unsupported numpy data type '{0}', kind '{1}'".format(
        col.dtype, col.dtype.kind))


def _map_spark_type(x) -> DataType:
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
    elif isinstance(x, pyspark.sql.types.BooleanType):
        return DataType.boolean
    elif isinstance(x, pyspark.sql.types.StringType):
        return DataType.string
    elif isinstance(x, pyspark.sql.types.BinaryType):
        return DataType.binary
    else:
        raise Exception("Unsupported Spark Type '{}', MLflow schema is only supported for scalar "
                        "Spark types.".format(type(x)))


def _is_spark_df(x) -> bool:
    try:
        return isinstance(x, importlib.import_module("pyspark.sql.dataframe").DataFrame)
    except ImportError:
        return False


def _infer_schema(data: MlflowModelDataset) -> Schema:
    if hasattr(data, "__len__") and len(data) == 0:
        return Schema([])
    if isinstance(data, dict):
        res = []
        for col in data.keys():
            ary = data[col]
            if not isinstance(ary, np.ndarray):
                raise TypeError("Data in the dictionary must be of type numpy.ndarray")
            dims = len(ary.shape)
            if dims == 1:
                res.append(ColSpec(col, _map_numpy_dtype(ary)))
            else:
                raise MlflowException("Data in the dictionary must be 1-dimensional, "
                                      "got shape {}".format(ary.shape))
        return Schema(res)

    if isinstance(data, pd.DataFrame):
        return Schema([ColSpec(col, _map_numpy_dtype(data[col].values)) for col in data.columns])
    elif isinstance(data, np.ndarray):
        if len(data.shape) > 2:
            raise MlflowException("Multidimensional arrays (aka tensors) are not supported, "
                                  "got array with shape {}".format(data.shape))
        if data.dtype == np.dtype("O"):
            df = pd.DataFrame(data).infer_objects()
            schema = _infer_schema(df)
            return Schema([ColSpec(None, t) for t in schema.column_types()])
        else:
            if len(data.shape) == 2:
                array_type = _map_numpy_dtype(data[0])
                return Schema([ColSpec(None, array_type)] * data.shape[1])
            else:
                array_type = _map_numpy_dtype(data)
                return Schema([ColSpec(None, array_type)])
    elif _is_spark_df(data):
        return Schema([ColSpec(field.name, _map_spark_type(field.dataType)) for field in
                       data.schema.fields])
    raise TypeError("Expected one of (pandas.DataFrame, numpy array, "
                    "dictionary of (name -> numpy.ndarray), pyspark.sql.DataFrame) "
                    "but got '{}'".format(type(data)))
