import base64

from json import JSONEncoder

from google.protobuf.json_format import MessageToJson, ParseDict
import numpy as np
import pandas as pd

from mlflow.types import DataType
from mlflow.types.schema import Schema


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def _stringify_all_experiment_ids(x):
    """Converts experiment_id fields which are defined as ints into strings in the given json.
    This is necessary for backwards- and forwards-compatibility with MLflow clients/servers
    running MLflow 0.9.0 and below, as experiment_id was changed from an int to a string.
    To note, the Python JSON serializer is happy to auto-convert strings into ints (so a
    server or client that sees the new format is fine), but is unwilling to convert ints
    to strings. Therefore, we need to manually perform this conversion.

    This code can be removed after MLflow 1.0, after users have given reasonable time to
    upgrade clients and servers to MLflow 0.9.1+.
    """
    if isinstance(x, dict):
        items = x.items()
        for k, v in items:
            if k == "experiment_id":
                x[k] = str(v)
            elif k == "experiment_ids":
                x[k] = [str(w) for w in v]
            elif k == "info" and isinstance(v, dict) and "experiment_id" in v and "run_uuid" in v:
                # shortcut for run info
                v["experiment_id"] = str(v["experiment_id"])
            elif k not in ("params", "tags", "metrics"):  # skip run data
                _stringify_all_experiment_ids(v)
    elif isinstance(x, list):
        for y in x:
            _stringify_all_experiment_ids(y)


def parse_dict(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JSON."""
    _stringify_all_experiment_ids(js_dict)
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)


class NumpyEncoder(JSONEncoder):
    """ Special json encoder for numpy types.
    Note that some numpy types doesn't have native python equivalence,
    hence json.dumps will raise TypeError.
    In this case, you'll need to convert your numpy types into its closest python equivalence.
    """

    def try_convert(self, o):
        def encode_binary(x):
            return base64.encodebytes(x).decode("ascii")

        if isinstance(o, np.ndarray):
            if o.dtype == np.object:
                return [self.try_convert(x)[0] for x in o.tolist()]
            elif o.dtype == np.bytes_:
                return np.vectorize(encode_binary)(o), True
            else:
                return o.tolist(), True

        if isinstance(o, np.generic):
            return o.item(), True
        if isinstance(o, bytes) or isinstance(o, bytearray):
            return encode_binary(o), True
        return o, False

    def default(self, o):  # pylint: disable=E0202
        res, converted = self.try_convert(o)
        if converted:
            return res
        else:
            return super().default(o)


def _dataframe_from_json(
    path_or_str, schema: Schema = None, pandas_orient: str = "split", precise_float=False
) -> pd.DataFrame:
    """
    Parse json into pandas.DataFrame. User can pass schema to ensure correct type parsing and to
    make any necessary conversions (e.g. string -> binary for binary columns).

    :param path_or_str: Path to a json file or a json string.
    :param schema: Mlflow schema used when parsing the data.
    :param pandas_orient: pandas data frame convention used to store the data.
    :return: pandas.DataFrame.
    """
    if schema is not None:
        dtypes = dict(zip(schema.column_names(), schema.pandas_types()))
        df = pd.read_json(
            path_or_str, orient=pandas_orient, dtype=dtypes, precise_float=precise_float
        )
        actual_cols = set(df.columns)
        for type_, name in zip(schema.column_types(), schema.column_names()):
            if type_ == DataType.binary and name in actual_cols:
                df[name] = df[name].map(lambda x: base64.decodebytes(bytes(x, "utf8")))
        return df
    else:
        return pd.read_json(
            path_or_str, orient=pandas_orient, dtype=False, precise_float=precise_float
        )


def _get_jsonable_obj(data, pandas_orient="records"):
    """Attempt to make the data json-able via standard library.
    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    :param data: data to be converted, works with pandas and numpy, rest will be returned as is.
    :param pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
                          dictionary using this Pandas serialization orientation.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient=pandas_orient)
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient=pandas_orient)
    else:  # by default just return whatever this is and hope for the best
        return data
