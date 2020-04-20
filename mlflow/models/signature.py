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


def save_example(path: str, input_example: ModelInputExample) -> str:
    """
    Save mlflow example into a file on a given path and return the resulting filename.

    The example(s) can be provided as :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray',
    python dictionary or python list. The assumption is that the example is conceptually either a
    DataFrame-like dataset or a numpy array or a dictionary of numpy arrays. Therefore, lists and
    dictionaries with no-numpy array values are converted to DataFrames first. This function will
    raise an Exception if such conversion fails.

    NOTE: If the input example is provided as a list, it is assumed that it is a list of rows. If
    you want to provide a single row example as a list, it is still expected as a list of lists.

    NOTE: Dictionaries with all scalar values are assumed ot be a single DataFrame row and will be
    converted accordingly.

    Example Storage Format
    ======================
    The examples are stored as json for portability and readability. Therefore, the contents of the
    example(s) must be jsonable. Mlflow will make the following conversions automatically on behalf
    of the user:

    - binary values: :py:class`bytes` or :py:class`bytearray` are converted to base64
      encoded strings.
    - numpy types: Numpy types are converted to the corresponding python types or their closest
      equivalent.

    The different input types are encoded as follows:

    - DataFrames are stored in pandas orient="split" format.
    - Lists are assumed to contain DataFrame rows and are converted into a DataFrame before writing
      them out.
    - Dictionaries are treated depending on their content:
        a) all values are numpy arrays - stored as json object with json arrays as values.
        b) else considered pandas DataFrame and is converted before writing out.
    - Numpy arrays are stored as json lists.

    :param path: Path where to store the example.
    :param input_example: Data with the input example(s).
    :return: Filename of the stored example.
    """
    if isinstance(input_example, dict):
        # assume dicts with non-numpy array values are dataframes
        if all([np.isscalar(x) for x in input_example.values()]):
            input_example = pd.DataFrame([input_example])
        elif any([not isinstance(x, np.ndarray) for x in input_example.values()]):
            input_example = pd.DataFrame.from_dict(input_example)

    if isinstance(input_example, list):
        input_example = pd.DataFrame(input_example)

    if isinstance(input_example, pd.DataFrame):
        example_filename = "input_dataframe_example.json"
        input_example = input_example.to_dict(orient="split")
    elif isinstance(input_example, dict):
        example_filename = "input_array_dictionary_example.json"
    elif isinstance(input_example, np.ndarray):
        example_filename = "input_array_example.json"
        input_example = input_example.tolist()
    else:
        raise TypeError("Unexpected type of input_example. Expected one of "
                        "(pandas.DataFrame, numpy.ndarray, dict, list), got {}".format(
            type(input_example)))
    with open(os.path.join(path, example_filename), "w") as f:
        json.dump(input_example, f, cls=NumpyEncoder)
    return example_filename


def _base64decode(x):
    return base64.decodebytes(x.encode("ascii"))


def read_example(path: str, schema: Schema = None) -> ModelInputExample:
    filename = os.path.split(path)[-1]
    if filename == "input_dataframe_example.json":
        return dataframe_from_json(path, schema, pandas_orient="split")
    with open(path, "r") as f:
        res = json.load(f)
    if filename == "input_array_example.json":
        if not isinstance(res, list):
            raise TypeError("Unexpected type, expected 'list', got {}".format(type(res)))

        if schema is not None:
            if len(set(schema.column_types())) > 1:
                res = np.array(res, dtype=np.object)
            else:
                res = np.array(res)
            binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]
            if binary_cols:
                convert = np.vectorize(_base64decode)
                for i in binary_cols:
                    res[..., i] = convert(res[..., i])
            return res
        else:
            return np.array(res)
    elif filename == "input_array_dictionary_example.json":
        if not isinstance(res, dict):
            raise TypeError("Unexpected type, expected 'dict', got {}".format(type(res)))
        if schema is not None:
            typemap = dict(zip(schema.column_names(), schema.column_types()))
            def convert(x, y):
                if typemap[x] == DataType.binary:
                    return np.array([_base64decode(z) for z in y], np.bytes_)
                else:
                    return np.array(y, dtype=typemap[x].to_numpy())

            return {x: convert(x, y) for x, y in res.items()}
        else:
            return {x: np.array(y) for x, y in res.items()}
    else:
        raise MlflowException("Unexpected example filename, expected one of ("
                              "'input_dataframe_example.json', "
                              "'input_array_dictionary_example.json', "
                              "'input_array_example.json'"
                              "), got '{}'".format(filename))


def dataframe_from_json(path_or_str, schema: Schema = None,
                        pandas_orient: str = "split") -> pd.DataFrame:
    """
    Read data frame back from json. User can pass schema to ensure correct type parsing and to make
    any necessary conversions (e.g. string -> binary for binary columns).

    :param path_or_str: Path to a json file or a json string.
    :param schema: Mlflow schema used when parsing the data.
    :param pandas_orient: pandas data frame convention used to store the data.
    :return: pandas.DataFrame.
    """
    if schema is not None:
        dtypes = dict(zip(schema.column_names(), schema.column_types()))
        df = pd.read_json(path_or_str, orient=pandas_orient, dtype=dtypes)[schema.column_names()]
        binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]

        for i in binary_cols:
            col = df.columns[i]
            df[col] = np.array(df[col].map(_base64decode), dtype=np.bytes_)
            return df
    else:
        return pd.read_json(path_or_str, orient=pandas_orient, dtype=False)


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
