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
from enum import Enum

import json
import numpy as np
import pandas as pd
from typing import List, Dict, TypeVar, Any

from mlflow.exceptions import MlflowException
from mlflow.models.utils import TensorsNotSupportedException


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

    def __repr__(self):
        return self.name

    def to_numpy(self) -> np.dtype:
        return self._numpy_type


class ColSpec(object):
    """
    Specification of a column used in model signature.
    Declares data type and optionally a name.
    """

    def __init__(self, type: DataType, name: str = None):  # pylint: disable=redefined-builtin
        self.name = name
        try:
            self.type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException("Unsupported type '{0}', expected instance of DataType or "
                                  "one of {1}".format(type, [t.name for t in DataType]))
        if not isinstance(self.type, DataType):
            raise TypeError("Expected mlflow.models.signature.Datatype or str for the 'type' "
                            "argument, but got {}".format(self.type.__class__))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a jsonable dictionary.
        :return: dictionary representation of the column spec.
        """
        return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        names_eq = (self.name is None and other.name is None) or self.name == other.name
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, signature_dict: Dict[str, Any]):
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


try:
    import pyspark.sql.dataframe

    MlflowModelDataset = TypeVar('MlflowModelDataset',
                                 pd.DataFrame,
                                 np.ndarray,
                                 Dict[str, np.ndarray],
                                 pyspark.sql.dataframe.DataFrame)
except ImportError:
    MlflowModelDataset = TypeVar('MlflowModelDataset',
                                 pd.DataFrame,
                                 np.ndarray,
                                 Dict[str, np.ndarray])


def infer_signature(model_input: MlflowModelDataset,
                    model_output: MlflowModelDataset = None) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).
    This method captures the column names and data types from the user data. The signature
    represents model input and output as dataframes with (optionally) named columns and data type
    specified as one of types defined in :py:class:`DataType`. This method will raise
    an exception if the user data contains incompatible types or is not passed in one of the
    supported formats (containers).

    The input should be one of these:
      - pandas.DataFrame
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType`.

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    :param model_input: Valid input to the model. E.g. (a subset of) the training dataset.
    :param model_output: Valid model output. E.g. Model predictions for the (subset of) training
                         dataset.
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    outputs = _infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)


def _infer_numpy_dtype(dtype: np.dtype) -> DataType:
    if not isinstance(dtype, np.dtype):
        raise TypeError("Expected numpy.dtype, got '{}'.".format(type(dtype)))
    if dtype.kind == "b":
        return DataType.boolean
    elif dtype.kind == "i" or dtype.kind == "u":
        if dtype.itemsize < 4 or (dtype.kind == "i" and dtype.itemsize == 4):
            return DataType.integer
        elif dtype.itemsize < 8 or (dtype.kind == "i" and dtype.itemsize == 8):
            return DataType.long
    elif dtype.kind == "f":
        if dtype.itemsize <= 4:
            return DataType.float
        elif dtype.itemsize <= 8:
            return DataType.double

    elif dtype.kind == "U":
        return DataType.string
    elif dtype.kind == "S":
        return DataType.binary
    elif dtype.kind == "O":
        raise Exception("Can not infer np.object without looking at the values, call "
                        "_map_numpy_array instead.")
    raise MlflowException("Unsupported numpy data type '{0}', kind '{1}'".format(
        dtype, dtype.kind))


def _infer_numpy_array(col: np.ndarray) -> DataType:
    if not isinstance(col, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(col)))
    if len(col.shape) > 1:
        raise MlflowException("Expected 1d array, got array with shape {}".format(col.shape))

    class IsInstanceOrNone(object):
        def __init__(self, *args):
            self.classes = args
            self.seen_instances = 0

        def __call__(self, x):
            if x is None:
                return True
            elif any(map(lambda c: isinstance(x, c), self.classes)):
                self.seen_instances += 1
                return True
            else:
                return False

    if col.dtype.kind == "O":
        is_binary_test = IsInstanceOrNone(bytes, bytearray)
        if all(map(is_binary_test, col)) and is_binary_test.seen_instances > 0:
            return DataType.binary
        is_string_test = IsInstanceOrNone(str)
        if all(map(is_string_test, col)) and is_string_test.seen_instances > 0:
            return DataType.string
        # NB: bool is also instance of int => boolean test must precede integer test.
        is_boolean_test = IsInstanceOrNone(bool)
        if all(map(is_boolean_test, col)) and is_boolean_test.seen_instances > 0:
            return DataType.boolean
        is_long_test = IsInstanceOrNone(int)
        if all(map(is_long_test, col)) and is_long_test.seen_instances > 0:
            return DataType.long
        is_double_test = IsInstanceOrNone(float)
        if all(map(is_double_test, col)) and is_double_test.seen_instances > 0:
            return DataType.double
        else:
            raise MlflowException("Unable to map 'np.object' type to MLflow DataType. np.object can"
                                  "be mapped iff all values have identical data type which is one "
                                  "of (string, (bytes or byterray),  int, float).")
    else:
        return _infer_numpy_dtype(col.dtype)


def _infer_spark_type(x) -> DataType:
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
        import pyspark.sql.dataframe
        return isinstance(x, pyspark.sql.dataframe.DataFrame)
    except ImportError:
        return False


def _infer_schema(data: MlflowModelDataset) -> Schema:
    if isinstance(data, dict):
        res = []
        for col in data.keys():
            ary = data[col]
            if not isinstance(ary, np.ndarray):
                raise TypeError("Data in the dictionary must be of type numpy.ndarray")
            dims = len(ary.shape)
            if dims == 1:
                res.append(ColSpec(type=_infer_numpy_array(ary), name=col))
            else:
                raise TensorsNotSupportedException("Data in the dictionary must be 1-dimensional, "
                                                   "got shape {}".format(ary.shape))
        return Schema(res)
    elif isinstance(data, pd.DataFrame):
        return Schema([ColSpec(type=_infer_numpy_array(data[col].values), name=col)
                       for col in data.columns])
    elif isinstance(data, np.ndarray):
        if len(data.shape) > 2:
            raise TensorsNotSupportedException("Attempting to infer schema from numpy array with "
                                               "shape {}".format(data.shape))
        if data.dtype == np.object:
            data = pd.DataFrame(data).infer_objects()
            return Schema([ColSpec(type=_infer_numpy_array(data[col].values))
                           for col in data.columns])
        if len(data.shape) == 1:
            return Schema([ColSpec(type=_infer_numpy_dtype(data.dtype))])
        elif len(data.shape) == 2:
            return Schema([ColSpec(type=_infer_numpy_dtype(data.dtype))] * data.shape[1])
    elif _is_spark_df(data):
        return Schema([ColSpec(type=_infer_spark_type(field.dataType), name=field.name)
                       for field in data.schema.fields])
    raise TypeError("Expected one of (pandas.DataFrame, numpy array, "
                    "dictionary of (name -> numpy.ndarray), pyspark.sql.DataFrame) "
                    "but got '{}'".format(type(data)))


def _base64decode(x):
    return base64.decodebytes(x.encode("ascii"))


def dataframe_from_json(path_or_str, schema: Schema = None,
                        pandas_orient: str = "split") -> pd.DataFrame:
    """
    Parse json into pandas.DataFrame. User can pass schema to ensure correct type parsing and to
    make any necessary conversions (e.g. string -> binary for binary columns).

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
