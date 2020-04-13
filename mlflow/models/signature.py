"""
The ``mlflow.models.signature`` module provides an API for specification of model signature and
of model example input. Both model signature and input example can be stored as part of MLflow
model metadata and provide valuable insights into the model behavior. In addition, the knowledge of
input and output model schema can be leveraged by MLflow deployment tools - for example spark_udf
can use the output schema to return result as struct (pandas.DataFrame).

**********************
Model Signature Format
**********************


**************************
Model Input Example Format
**************************


Creating Schema


The model schema can be constructed by hand or inferred from user provided dataset using
:py:mod:`mlflow.models.signature.infer_signature`.


"""
import base64
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
    float = (4, np.float)
    double = (5, np.double)
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

    def __init__(self, name: str, type: DataType):
        self.name = name
        try:
            self.type = getattr(DataType, type) if isinstance(type, str) else type
        except AttributeError:
            raise MlflowException("Unsupported type '{0}', expected instance of DataType or "
                                  "one of {1}".format(type, [t.name for t in DataType]))

    def to_dict(self):
        """
        Serialize into a jsonable dictionary.
        :return: dictionary representation of the column spec.
        """
        return {"name": self.name, "type": self.type.name}

    def __eq__(self, other):
        names_eq = self.name is None and other.name is None or self.name == other.name
        return names_eq and self.type == other.type

    def __str__(self):
        return "{name}: {type}".format(name=self.name, type=self.type)

    def __repr__(self):
        return "{name}: {type}".format(name=self.name, type=self.type)


class Schema(object):
    def __init__(self, cols: List[ColSpec]):
        self._cols = cols

    @property
    def columns(self):
        return self._cols

    def column_names(self):
        return [x.name or i for i, x in enumerate(self._cols)]

    def column_types(self):
        return [x.type for x in self._cols]

    def numpy_types(self):
        return [x.type.to_numpy() for x in self._cols]

    def to_json(self):
        return json.dumps([x.to_dict() for x in self._cols])

    def __eq__(self, other):
        if isinstance(other, Schema):
            return self.columns == other.columns
        else:
            return False

    def __repr__(self):
        return repr(self.columns)

    @classmethod
    def from_json(cls, json_str):
        return cls([ColSpec(**x) for x in json.loads(json_str)])


class ModelSignature(object):
    """
    ModelSignature specifies schema of model's inputs and outputs.

    The current supported schema for both the input and the output is a data-frame like schema
    defined as a list of column specification ``ColSpec``. Columns can be named and must specify
    their data type. Currently the list of supported types is limited to scalar data types defined
    in ``DataType`` enum.

    ModelSignature can be inferred from training dataset and model predictions using
    ``mlflow.models.signature.infer_signature``, or alternatively constructed by hand by passing a
    lists of input and output Column specifications.
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

    def to_dict(self):
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that when the
        representation is compact when embedded in a MLmofel yaml file.


        :return: dictionary representation with input and output shcema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None
        }

    @classmethod
    def from_dict(cls, signature_dict):
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

    def __eq__(self, other):
        return self.inputs == other.inputs and self.outputs == other.outputs

    def __repr__(self):
        import json
        return json.dumps({"ModelSignature": self.to_dict()}, indent=2)


ModelInputExample = TypeVar('InputExample', pd.DataFrame, np.ndarray, dict, list)
MlflowModelDataset = TypeVar('MlflowModelDataset', pd.DataFrame, np.ndarray, Dict[str, np.ndarray])


def infer_signature(model_input: MlflowModelDataset,
                    model_output: MlflowModelDataset = None) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).
    This method captures the column names and data types from the user data. The signature
    represents model input and output as dataframes with (optionally) named columns and data type
    specified as one of types defined in mlflow.models.signature.DataType. This method will raise
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
    :param model_output: Valid model output. model predictions on the (subset of) training dataset.
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    outputs = _infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)


def save_example(path: str, input_example: ModelInputExample, input_schema: Schema = None):
    if isinstance(input_example, dict):
        input_example = _dataframe_from_dict(input_example, input_schema)
    elif isinstance(input_example, list):
        input_example = pd.DataFrame(input_example)
    if isinstance(input_example, pd.DataFrame):
        example_filename = "input_dataframe_example.json"
    elif isinstance(input_example, np.ndarray):
        example_filename = "input_array_example.json"
    else:
        raise TypeError("Unexpected type of input_example. Expected one of "
                        "(pandas.DataFrame, numpy.ndarray, dict, list), got {}".format(
            type(input_example)))
    print()
    print(input_example)
    print()
    print(input_schema)
    print()
    with open(os.path.join(path, example_filename), "w") as f:
        to_json(input_example, pandas_orient="records", schema=input_schema, output_stream=f)
    return example_filename


def from_json(json_str, schema: Schema = None, pandas_orient="records"):
    if schema is not None:
        dtypes = dict(zip(schema.column_names(), schema.column_types()))
        df = pd.read_json(json_str, orient=pandas_orient, dtype=dtypes)
        binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]

        def base64decode(x):
            return base64.decodebytes(x.encode("ascii"))

        for i in binary_cols:
            col = df.columns[i]
            df[col] = np.array(df[col].map(base64decode), dtype=np.bytes_)
            return df
    else:
        return pd.read_json(json_str, orient=pandas_orient, dtype=False)


def to_json(data: MlflowModelDataset, pandas_orient="records", schema: Schema = None,
            output_stream=None):
    """Attempt to make the data json-able via standard library.
    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    :param data: data to be converted, works with pandas and numpy, rest will be returned as is.
    :param pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
                          dictionary using this Pandas serialization orientation.
    """

    def get_jsonable_data(data: MlflowModelDataset, pandas_orient, schema: Schema = None):
        if schema is not None:
            binary_cols = [i for i, x in enumerate(schema.column_types()) if x == DataType.binary]
        else:
            binary_cols = []

        print("binary cols  = ", binary_cols)
        def base64encode(x):
            return base64.encodebytes(x).decode("ascii")

        def base64_encode_ndarray(x, binary_cols):
            print("base64 encoding cols {}".format(binary_cols))
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
            return data.to_dict(orient=pandas_orient)

        if isinstance(data, dict):
            if binary_cols:
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
                return new_data
        if isinstance(data, np.ndarray):
            if binary_cols:
                data = base64_encode_ndarray(data, binary_cols)
            return data.tolist()
        else:  # by default just return whatever this is and hope for the best
            return data

    if output_stream is not None:
        json.dump(get_jsonable_data(data, pandas_orient, schema), output_stream, cls=NumpyEncoder)
    else:
        return json.dumps(get_jsonable_data(data, pandas_orient), cls=NumpyEncoder)


def _dataframe_from_dict(d: dict, schema: Schema = None) -> pd.DataFrame:
    print()
    print("dict")
    print(d)
    print("values")
    print(d.values())
    print("====")
    print()
    if all([np.isscalar(x) for x in d.values()]):
        d = {x: np.array([v], dtype="object") for x, v in d.items()}
    if schema is not None:
        dtypes = dict(zip(schema.column_names(), schema.numpy_types()))
        d = {x: y.astype(dtype=dtypes[x]) for x, y in d.items()}
        return pd.DataFrame.from_dict(d)
    else:
        d = {x: y.astype("object") for x, y in d.items()}
        print(d)
        return pd.DataFrame.from_dict(d).infer_objects()


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
    import pyspark.sql.dataframe
    return isinstance(x, pyspark.sql.dataframe.DataFrame)


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
            # elif dims == 2 and ary.dtype == np.uint8:
            #     res.append(ColSpec(col, DataType.binary))
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
