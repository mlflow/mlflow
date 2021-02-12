import base64

from json import JSONEncoder

from google.protobuf.json_format import MessageToJson, ParseDict

from mlflow.exceptions import MlflowException
from collections import defaultdict


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
        import numpy as np

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
    path_or_str, schema=None, pandas_orient: str = "split", precise_float=False
):
    """
    Parse json into pandas.DataFrame. User can pass schema to ensure correct type parsing and to
    make any necessary conversions (e.g. string -> binary for binary columns).

    :param path_or_str: Path to a json file or a json string.
    :param schema: Mlflow schema used when parsing the data.
    :param pandas_orient: pandas data frame convention used to store the data.
    :return: pandas.DataFrame.
    """
    import pandas as pd

    from mlflow.types import DataType

    if schema is not None:
        dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
        df = pd.read_json(
            path_or_str, orient=pandas_orient, dtype=dtypes, precise_float=precise_float
        )
        actual_cols = set(df.columns)
        for type_, name in zip(schema.input_types(), schema.input_names()):
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
    import numpy as np
    import pandas as pd

    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient=pandas_orient)
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient=pandas_orient)
    else:  # by default just return whatever this is and hope for the best
        return data


def parse_tf_serving_input(inp_dict, schema = None):
    """
    :param inp_dict: A dict deserialized from a JSON string formatted as described in TF's
                     serving API doc
                     (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
    """
    import numpy as np

    # pylint: disable=broad-except
    if "signature_name" in inp_dict:
        raise MlflowException(
            'Failed to parse data as TF serving input. "signature_name" is currently'
            " not supported."
        )

    if not (list(inp_dict.keys()) == ["instances"] or list(inp_dict.keys()) == ["inputs"]):
        raise MlflowException(
            'Failed to parse data as TF serving input. One of "instances" and'
            ' "inputs" must be specified (not both or any other keys).'
        )

    # Read the JSON
    try:
        if "instances" in inp_dict:
            items = inp_dict["instances"]
            if len(items) > 0 and isinstance(items[0], dict):
                # convert items to column format (map column/input name to tensor)
                data = defaultdict(list)
                for item in items:
                    for k, v in item.items():
                        data[k].append(v)
                data = {k: np.array(v) for k, v in data.items()}
            else:
                data = np.array(items)
        else:
            # items already in column format, convert values to tensor
            items = inp_dict["inputs"]
            if len(items) > 0 and isinstance(items, dict):
                data = {k: np.array(v) for k, v in items.items()}
            else:
                data = np.array(items)
    except Exception:
        raise MlflowException(
            "Failed to parse data as TF serving input. Ensure that the input is"
            " a valid JSON-formatted string that conforms to the request body for"
            " TF serving's Predict API as documented at"
            " https://www.tensorflow.org/tfx/serving/api_rest#request_format_2"
        )

    # Sanity check inputted data
    if isinstance(data, dict):
        # ensure all columns have the same number of items
        expected_len = len(list(data.values())[0])
        if not all(len(v) == expected_len for v in data.values()):
            raise MlflowException(
                "Failed to parse data as TF serving input. The length of values for"
                " each input/column name are not the same"
            )

    # Attempt casting
    if schema is not None:
        if schema.has_input_names():
            input_names = schema.input_names()
            if len(input_names) == 1 and isinstance(data, np.ndarray):
                # for schemas with a single column, match input with column
                data = {input_names[0]: data}
            for col_name, tensor_spec in zip(schema.input_names(), schema.inputs):
                data[col_name] = data[col_name].astype(tensor_spec.type, casting="unsafe")
        else:
            data = data.astype(schema.inputs[0].type, casting="unsafe")
    return data
