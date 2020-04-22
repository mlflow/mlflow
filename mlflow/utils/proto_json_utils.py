import base64
from json import JSONEncoder

from google.protobuf.json_format import MessageToJson, ParseDict
import numpy as np


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
    def default(self, o):  # pylint: disable=E0202
        def encode_binary(x):
            return base64.encodebytes(x).decode("ascii")

        if isinstance(o, np.ndarray):
            if o.dtype == np.object:
                return [self.default(x) for x in o.tolist()]
            elif o.dtype == np.bytes_:
                return np.vectorize(encode_binary)(o)
            else:
                return o.tolist()

        if isinstance(o, np.generic):
            return np.asscalar(o)
        if isinstance(o, bytes) or isinstance(o, bytearray):
            return encode_binary(o)
        return o
