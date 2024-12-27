import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Optional

import pydantic
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict

from mlflow.exceptions import MlflowException
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

_PROTOBUF_INT64_FIELDS = [
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_FIXED64,
    FieldDescriptor.TYPE_SFIXED64,
    FieldDescriptor.TYPE_SINT64,
]

from mlflow.protos.databricks_pb2 import BAD_REQUEST


def _mark_int64_fields_for_proto_maps(proto_map, value_field_type):
    """Converts a proto map to JSON, preserving only int64-related fields."""
    json_dict = {}
    for key, value in proto_map.items():
        # The value of a protobuf map can only be a scalar or a message (not a map or repeated
        # field).
        if value_field_type == FieldDescriptor.TYPE_MESSAGE:
            json_dict[key] = _mark_int64_fields(value)
        elif value_field_type in _PROTOBUF_INT64_FIELDS:
            json_dict[key] = int(value)
        elif isinstance(key, int):
            json_dict[key] = value
    return json_dict


def _mark_int64_fields(proto_message):
    """Converts a proto message to JSON, preserving only int64-related fields."""
    json_dict = {}
    for field, value in proto_message.ListFields():
        if (
            # These three conditions check if this field is a protobuf map.
            # See the official implementation: https://bit.ly/3EMx1rl
            field.type == FieldDescriptor.TYPE_MESSAGE
            and field.message_type.has_options
            and field.message_type.GetOptions().map_entry
        ):
            # Deal with proto map fields separately in another function.
            json_dict[field.name] = _mark_int64_fields_for_proto_maps(
                value, field.message_type.fields_by_name["value"].type
            )
            continue

        if field.type == FieldDescriptor.TYPE_MESSAGE:
            ftype = partial(_mark_int64_fields)
        elif field.type in _PROTOBUF_INT64_FIELDS:
            ftype = int
        else:
            # Skip all non-int64 fields.
            continue

        json_dict[field.name] = (
            [ftype(v) for v in value]
            if field.label == FieldDescriptor.LABEL_REPEATED
            else ftype(value)
        )
    return json_dict


def _merge_json_dicts(from_dict, to_dict):
    """Merges the json elements of from_dict into to_dict. Only works for json dicts
    converted from proto messages
    """
    for key, value in from_dict.items():
        if isinstance(key, int) and str(key) in to_dict:
            # When the key (i.e. the proto field name) is an integer, it must be a proto map field
            # with integer as the key. For example:
            # from_dict is {'field_map': {1: '2', 3: '4'}}
            # to_dict is {'field_map': {'1': '2', '3': '4'}}
            # So we need to replace the str keys with int keys in to_dict.
            to_dict[key] = to_dict[str(key)]
            del to_dict[str(key)]

        if key not in to_dict:
            continue

        if isinstance(value, dict):
            _merge_json_dicts(from_dict[key], to_dict[key])
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    _merge_json_dicts(v, to_dict[key][i])
                else:
                    to_dict[key][i] = v
        else:
            to_dict[key] = from_dict[key]
    return to_dict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""

    # Google's MessageToJson API converts int64 proto fields to JSON strings.
    # For more info, see https://github.com/protocolbuffers/protobuf/issues/2954
    json_dict_with_int64_as_str = json.loads(
        MessageToJson(message, preserving_proto_field_name=True)
    )
    # We convert this proto message into a JSON dict where only int64 proto fields
    # are preserved, and they are treated as JSON numbers, not strings.
    json_dict_with_int64_fields_only = _mark_int64_fields(message)
    # By merging these two JSON dicts, we end up with a JSON dict where int64 proto fields are not
    # converted to JSON strings. Int64 keys in proto maps will always be converted to JSON strings
    # because JSON doesn't support non-string keys.
    json_dict_with_int64_as_numbers = _merge_json_dicts(
        json_dict_with_int64_fields_only, json_dict_with_int64_as_str
    )
    return json.dumps(json_dict_with_int64_as_numbers, indent=2)


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
    """Special json encoder for numpy types.
    Note that some numpy types doesn't have native python equivalence,
    hence json.dumps will raise TypeError.
    In this case, you'll need to convert your numpy types into its closest python equivalence.
    """

    def try_convert(self, o):
        import numpy as np
        import pandas as pd

        def encode_binary(x):
            return base64.encodebytes(x).decode("ascii")

        if isinstance(o, np.ndarray):
            if o.dtype == object:
                return [self.try_convert(x)[0] for x in o.tolist()], True
            elif o.dtype == np.bytes_:
                return np.vectorize(encode_binary)(o), True
            else:
                return o.tolist(), True

        if isinstance(o, np.generic):
            return o.item(), True
        if isinstance(o, (bytes, bytearray)):
            return encode_binary(o), True
        if isinstance(o, np.datetime64):
            return np.datetime_as_string(o), True
        if isinstance(o, (pd.Timestamp, datetime.date, datetime.datetime, datetime.time)):
            return o.isoformat(), True
        if isinstance(o, pydantic.BaseModel):
            return o.model_dump() if IS_PYDANTIC_V2_OR_NEWER else o.dict(), True
        return o, False

    def default(self, o):
        res, converted = self.try_convert(o)
        if converted:
            return res
        else:
            return super().default(o)


class MlflowInvalidInputException(MlflowException):
    def __init__(self, message):
        super().__init__(f"Invalid input. {message}", error_code=BAD_REQUEST)


class MlflowFailedTypeConversion(MlflowInvalidInputException):
    def __init__(self, col_name, col_type, ex):
        super().__init__(
            message=f"Data is not compatible with model signature. "
            f"Failed to convert column {col_name} to type '{col_type}'. Error: '{ex!r}'"
        )


def cast_df_types_according_to_schema(pdf, schema):
    import numpy as np

    from mlflow.models.utils import _enforce_array, _enforce_map, _enforce_object
    from mlflow.types.schema import AnyType, Array, DataType, Map, Object

    actual_cols = set(pdf.columns)
    if schema.has_input_names():
        dtype_list = zip(schema.input_names(), schema.input_types())
    elif schema.is_tensor_spec() and len(schema.input_types()) == 1:
        dtype_list = zip(actual_cols, [schema.input_types()[0] for _ in actual_cols])
    else:
        n = min(len(schema.input_types()), len(pdf.columns))
        dtype_list = zip(pdf.columns[:n], schema.input_types()[:n])
    required_input_names = set(schema.required_input_names())

    for col_name, col_type_spec in dtype_list:
        if isinstance(col_type_spec, DataType):
            col_type = col_type_spec.to_pandas()
        else:
            col_type = col_type_spec
        if col_name in actual_cols:
            required = col_name in required_input_names
            try:
                if isinstance(col_type_spec, DataType) and col_type_spec == DataType.binary:
                    # NB: We expect binary data to be passed base64 encoded
                    pdf[col_name] = pdf[col_name].map(
                        lambda x: base64.decodebytes(bytes(x, "utf8"))
                    )
                elif col_type == np.dtype(bytes):
                    pdf[col_name] = pdf[col_name].map(lambda x: bytes(x, "utf8"))
                elif schema.is_tensor_spec() and isinstance(pdf[col_name].iloc[0], list):
                    # For dataframe with multidimensional column, it contains
                    # list type values, we cannot convert
                    # its type by `astype`, skip conversion.
                    # The conversion will be done in `_enforce_schema` while
                    # `PyFuncModel.predict` being called.
                    pass
                elif isinstance(col_type_spec, Array):
                    pdf[col_name] = pdf[col_name].map(
                        lambda x: _enforce_array(x, col_type_spec, required=required)
                    )
                elif isinstance(col_type_spec, Object):
                    pdf[col_name] = pdf[col_name].map(
                        lambda x: _enforce_object(x, col_type_spec, required=required)
                    )
                elif isinstance(col_type_spec, Map):
                    pdf[col_name] = pdf[col_name].map(
                        lambda x: _enforce_map(x, col_type_spec, required=required)
                    )
                elif isinstance(col_type_spec, AnyType):
                    pass
                else:
                    pdf[col_name] = pdf[col_name].astype(col_type, copy=False)
            except Exception as ex:
                raise MlflowFailedTypeConversion(col_name, col_type, ex)
    return pdf


def dataframe_from_parsed_json(decoded_input, pandas_orient, schema=None):
    """Convert parsed json into pandas.DataFrame. If schema is provided this methods will attempt to
    cast data types according to the schema. This include base64 decoding for binary columns.

    Args:
        decoded_input: Parsed json - either a list or a dictionary.
        pandas_orient: pandas data frame convention used to store the data.
        schema: MLflow schema used when parsing the data.

    Returns:
        pandas.DataFrame.
    """
    import pandas as pd

    if pandas_orient == "records":
        if not isinstance(decoded_input, list):
            if isinstance(decoded_input, dict):
                typemessage = "dictionary"
            else:
                typemessage = f"type {type(decoded_input)}"
            raise MlflowInvalidInputException(
                f"Dataframe records format must be a list of records. Got {typemessage}."
            )
        try:
            pdf = pd.DataFrame(data=decoded_input)
        except Exception as ex:
            raise MlflowInvalidInputException(
                f"Provided dataframe_records field is not a valid dataframe representation in "
                f"'records' format. Error: '{ex}'"
            )
    elif pandas_orient == "split":
        if not isinstance(decoded_input, dict):
            if isinstance(decoded_input, list):
                typemessage = "list"
            else:
                typemessage = f"type {type(decoded_input)}"
            raise MlflowInvalidInputException(
                f"Dataframe split format must be a dictionary. Got {typemessage}."
            )
        keys = set(decoded_input.keys())
        missing_data = "data" not in keys
        extra_keys = keys.difference({"columns", "data", "index"})
        if missing_data or extra_keys:
            raise MlflowInvalidInputException(
                f"Dataframe split format must have 'data' field and optionally 'columns' "
                f"and 'index' fields. Got {keys}.'"
            )
        try:
            pdf = pd.DataFrame(
                index=decoded_input.get("index"),
                columns=decoded_input.get("columns"),
                data=decoded_input["data"],
            )
        except Exception as ex:
            raise MlflowInvalidInputException(
                f"Provided dataframe_split field is not a valid dataframe representation in "
                f"'split' format. Error: '{ex}'"
            )
    if schema is not None:
        pdf = cast_df_types_according_to_schema(pdf, schema)
    return pdf


def dataframe_from_raw_json(path_or_str, schema=None, pandas_orient: str = "split"):
    """Parse raw json into a pandas.Dataframe.

    If schema is provided this methods will attempt to cast data types according to the schema. This
    include base64 decoding for binary columns.

    Args:
        path_or_str: Path to a json file or a json string.
        schema: MLflow schema used when parsing the data.
        pandas_orient: pandas data frame convention used to store the data.

    Returns:
        pandas.DataFrame.
    """
    if os.path.exists(path_or_str):
        with open(path_or_str) as f:
            parsed_json = json.load(f)
    else:
        parsed_json = json.loads(path_or_str)

    return dataframe_from_parsed_json(parsed_json, pandas_orient, schema)


def _get_jsonable_obj(data, pandas_orient="records"):
    """Attempt to make the data json-able via standard library.

    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    Args:
        data: Data to be converted, works with pandas and numpy, rest will be returned as is.
        pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
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


def convert_data_type(data, spec):
    """
    Convert input data to the type specified in the spec.

    Args:
        data: Input data.
        spec: ColSpec or TensorSpec.
    """
    import numpy as np

    from mlflow.models.utils import _enforce_array, _enforce_map, _enforce_object
    from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, TensorSpec

    try:
        if spec is None:
            return np.array(data)
        if isinstance(spec, TensorSpec):
            return np.array(data, dtype=spec.type)
        if isinstance(spec, ColSpec):
            if isinstance(spec.type, DataType):
                return (
                    np.array(data, spec.type.to_numpy())
                    if isinstance(data, (list, np.ndarray))
                    else np.array([data], spec.type.to_numpy())[0]
                )
            elif isinstance(spec.type, Array):
                # convert to numpy array for backwards compatibility
                return np.array(_enforce_array(data, spec.type, required=spec.required))
            elif isinstance(spec.type, Object):
                return _enforce_object(data, spec.type, required=spec.required)
            elif isinstance(spec.type, Map):
                return _enforce_map(data, spec.type, required=spec.required)
            elif isinstance(spec.type, AnyType):
                return data
    except MlflowException as e:
        raise MlflowInvalidInputException(e.message)
    except Exception as ex:
        raise MlflowInvalidInputException(f"{ex}")

    raise MlflowInvalidInputException(
        f"Failed to convert data type for data `{data}` with spec `{spec}`."
    )


def _cast_schema_type(input_data, schema=None):
    import numpy as np

    input_data = deepcopy(input_data)
    # spec_name -> spec mapping
    types_dict = schema.input_dict() if schema and schema.has_input_names() else {}
    if schema is not None:
        if (
            len(types_dict) == 1
            and isinstance(input_data, list)
            and not any(isinstance(x, dict) for x in input_data)
        ):
            # for data with a single column (not List[Dict]), match input with column
            input_data = {next(iter(types_dict)): input_data}
        # Un-named schema should only contain a single column or a single value
        elif not schema.has_input_names() and not (
            isinstance(input_data, list) or np.isscalar(input_data)
        ):
            raise MlflowInvalidInputException(
                "Failed to parse input data. This model contains an un-named "
                " model signature which expects a single n-dimensional array or "
                "a single value as input, however, an input of type "
                f"{type(input_data)} was found."
            )
    if isinstance(input_data, dict):
        # each key corresponds to a column, values should be
        # checked against the schema
        input_data = {
            col: convert_data_type(data, types_dict.get(col)) for col, data in input_data.items()
        }
    elif isinstance(input_data, list):
        # List of dictionaries of column_name -> value mapping
        # List[Dict] must correspond to a schema with named columns
        if all(isinstance(x, dict) for x in input_data):
            input_data = [
                {col: convert_data_type(value, types_dict.get(col)) for col, value in data.items()}
                for data in input_data
            ]
        # List of values
        else:
            spec = schema.inputs[0] if schema else None
            input_data = convert_data_type(input_data, spec)
    else:
        spec = schema.inputs[0] if schema else None
        try:
            input_data = convert_data_type(input_data, spec)
        except Exception as e:
            raise MlflowInvalidInputException(
                f"Failed to convert data `{input_data}` to type `{spec}` defined "
                "in the model signature."
            ) from e
    return input_data


def parse_instances_data(data, schema=None):
    import numpy as np

    from mlflow.types.schema import Array

    if "instances" not in data:
        raise MlflowInvalidInputException("Expecting data to have `instances` as key.")
    data = data["instances"]
    # List[Dict]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # convert items to column format (map column/input name to tensor)
        data_dict = defaultdict(list)
        types_dict = schema.input_dict() if schema and schema.has_input_names() else {}
        for item in data:
            for col, v in item.items():
                data_dict[col].append(convert_data_type(v, types_dict.get(col)))
        # convert to numpy array for backwards compatibility
        data = {col: np.array(v) for col, v in data_dict.items()}
    else:
        data = _cast_schema_type(data, schema)

    # Sanity check inputted data. This check will only be applied
    # when the row-format `instances` is used since it requires
    # same 0-th dimension for all items.
    if isinstance(data, dict):
        # ensure all columns have the same number of items
        # Only check the data when it's a list or numpy array
        check_data = {k: v for k, v in data.items() if isinstance(v, (list, np.ndarray))}
        if schema and schema.has_input_names():
            # Only check required columns
            required_cols = schema.required_input_names()
            # For Array schema we should not check the length of the data matching
            check_cols = {
                col for col, spec in schema.input_dict().items() if not isinstance(spec.type, Array)
            }
            check_cols = list(set(required_cols) & check_cols & set(check_data.keys()))
        else:
            check_cols = list(check_data.keys())

        if check_cols:
            expected_len = len(check_data[check_cols[0]])
            if not all(len(check_data[col]) == expected_len for col in check_cols[1:]):
                raise MlflowInvalidInputException(
                    "The length of values for each input/column name are not the same"
                )
    return data


# TODO: Reuse this function for `inputs` key data parsing in serving, and
# add `convert_to_numpy` param to avoid converting data to numpy arrays for
# genAI flavors.
def parse_inputs_data(inputs_data_or_path, schema=None):
    """
    Helper function to cast inputs_data based on the schema.
    Inputs data must be able to pass to the model for pyfunc predict directly.

    Args:
        inputs_data_or_path: A json-serializable object or path to a json file
        schema: data schema to cast to. Be of type `mlflow.types.Schema`.
    """
    if isinstance(inputs_data_or_path, str) and os.path.exists(inputs_data_or_path):
        with open(inputs_data_or_path) as handle:
            inputs_data = json.load(handle)
    else:
        inputs_data = inputs_data_or_path
    return _cast_schema_type(inputs_data, schema)


def parse_tf_serving_input(inp_dict, schema=None):
    """
    Args:
        inp_dict: A dict deserialized from a JSON string formatted as described in TF's
            serving API doc
            (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
        schema: MLflow schema used when parsing the data.
    """

    if "signature_name" in inp_dict:
        raise MlflowInvalidInputException('"signature_name" parameter is currently not supported')

    if not (list(inp_dict.keys()) == ["instances"] or list(inp_dict.keys()) == ["inputs"]):
        raise MlflowInvalidInputException(
            'One of "instances" and "inputs" must be specified (not both or any other keys).'
            f"Received: {list(inp_dict.keys())}"
        )

    # Read the JSON
    try:
        # objects & arrays schema for List[Dict] and Dict[List] are different
        # so the conversion for `instances` below changes the schema.
        # e.g.
        # [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}] -> {"col1": [1, 3], "col2": [2, 4]}
        # Schema([ColSpec(long, "col1"), ColSpec(long, "col2")]) ->
        # Schema([ColSpec(Array(long), "col1"), ColSpec(Array(long), "col2")])
        # To avoid this, we shouldn't use `instances` for such data.
        if "instances" in inp_dict:
            return parse_instances_data(inp_dict, schema)
        else:
            # items already in column format, convert values to tensor
            return _cast_schema_type(inp_dict["inputs"], schema)
    except MlflowException as e:
        raise e
    except Exception as e:
        # Add error into message to provide details for serving usage
        raise MlflowInvalidInputException(
            f"Ensure that the input is a valid JSON-formatted string.\nError: {e!r}"
        ) from e


# Reference: https://stackoverflow.com/a/12126976
class _CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        import pandas as pd

        if isinstance(o, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
            return o.isoformat()

        if isinstance(o, np.ndarray):
            return o.tolist()

        return super().default(o)


def get_jsonable_input(name, data):
    import numpy as np

    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        raise MlflowException(f"Incompatible input type:{type(data)} for input {name}.")


def dump_input_data(data, inputs_key="inputs", params: Optional[dict[str, Any]] = None):
    """
    Args:
        data: Input data.
        inputs_key: Key to represent data in the request payload.
        params: Additional parameters to pass to the model for inference.
    """
    import numpy as np
    import pandas as pd

    # Convert scipy data to numpy array
    if importlib.util.find_spec("scipy.sparse"):
        from scipy.sparse import csc_matrix, csr_matrix

        if isinstance(data, (csc_matrix, csr_matrix)):
            data = data.toarray()

    if isinstance(data, pd.DataFrame):
        post_data = {"dataframe_split": data.to_dict(orient="split")}
    elif isinstance(data, dict):
        post_data = {inputs_key: {k: get_jsonable_input(k, v) for k, v in data}}
    elif isinstance(data, np.ndarray):
        post_data = {inputs_key: data.tolist()}
    elif isinstance(data, list):
        post_data = {inputs_key: data}
    else:
        post_data = data

    if params is not None:
        if not isinstance(params, dict):
            raise MlflowException(
                f"Params must be a dictionary. Got type '{type(params).__name__}'."
            )
        # if post_data is not dictionary, params should be included in post_data directly
        if isinstance(post_data, dict):
            post_data["params"] = params

    if not isinstance(post_data, str):
        post_data = json.dumps(post_data, cls=_CustomJsonEncoder)

    return post_data
