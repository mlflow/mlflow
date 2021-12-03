"""
Scoring server for python model format.
The passed int model is expected to have function:
   predict(pandas.Dataframe) -> pandas.DataFrame

Input, expected in text/csv or application/json format,
is parsed into pandas.DataFrame and passed to the model.

Defines two endpoints:
    /ping used for health check
    /invocations used for scoring
"""
from collections import OrderedDict
from typing import Tuple, Dict
import flask
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import traceback

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException
from mlflow.types import Schema
from mlflow.utils import reraise
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.proto_json_utils import (
    NumpyEncoder,
    _dataframe_from_json,
    _get_jsonable_obj,
    parse_tf_serving_input,
)

try:
    from mlflow.pyfunc import load_model, PyFuncModel
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.server.handlers import catch_mlflow_exception

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

_SERVER_MODEL_PATH = "__pyfunc_model_path__"

CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_JSON_RECORDS_ORIENTED = "application/json; format=pandas-records"
CONTENT_TYPE_JSON_SPLIT_ORIENTED = "application/json; format=pandas-split"
CONTENT_TYPE_JSON_SPLIT_NUMPY = "application/json-numpy-split"

CONTENT_TYPES = [
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_JSON_SPLIT_NUMPY,
]

CONTENT_TYPE_FORMAT_RECORDS_ORIENTED = "pandas-records"
CONTENT_TYPE_FORMAT_SPLIT_ORIENTED = "pandas-split"

FORMATS = [CONTENT_TYPE_FORMAT_RECORDS_ORIENTED, CONTENT_TYPE_FORMAT_SPLIT_ORIENTED]

PREDICTIONS_WRAPPER_ATTR_NAME_ENV_KEY = "PREDICTIONS_WRAPPER_ATTR_NAME"

_logger = logging.getLogger(__name__)


def infer_and_parse_json_input(json_input, schema: Schema = None):
    """
    :param json_input: A JSON-formatted string representation of TF serving input or a Pandas
                       DataFrame, or a stream containing such a string representation.
    :param schema: Optional schema specification to be used during parsing.
    """
    try:
        decoded_input = json.loads(json_input)
    except json.decoder.JSONDecodeError:
        _handle_serving_error(
            error_message=(
                "Failed to parse input from JSON. Ensure that input is a valid JSON"
                " formatted string."
            ),
            error_code=BAD_REQUEST,
        )

    if isinstance(decoded_input, list):
        return parse_json_input(json_input=json_input, orient="records", schema=schema)
    elif isinstance(decoded_input, dict):
        if "instances" in decoded_input or "inputs" in decoded_input:
            try:
                return parse_tf_serving_input(decoded_input, schema=schema)
            except MlflowException as ex:
                _handle_serving_error(
                    error_message=(ex.message),
                    error_code=BAD_REQUEST,
                )
        else:
            return parse_json_input(json_input=json_input, orient="split", schema=schema)
    else:
        _handle_serving_error(
            error_message=(
                "Failed to parse input from JSON. Ensure that input is a valid JSON"
                " list or dictionary."
            ),
            error_code=BAD_REQUEST,
        )


def parse_json_input(json_input, orient="split", schema: Schema = None):
    """
    :param json_input: A JSON-formatted string representation of a Pandas DataFrame, or a stream
                       containing such a string representation.
    :param orient: The Pandas DataFrame orientation of the JSON input. This is either 'split'
                   or 'records'.
    :param schema: Optional schema specification to be used during parsing.
    """

    try:
        return _dataframe_from_json(json_input, pandas_orient=orient, schema=schema)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid JSON-formatted Pandas DataFrame with the `{orient}` orient"
                " produced using the `pandas.DataFrame.to_json(..., orient='{orient}')`"
                " method.".format(orient=orient)
            ),
            error_code=BAD_REQUEST,
        )


def parse_csv_input(csv_input):
    """
    :param csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                      containing such a string representation.
    """

    try:
        return pd.read_csv(csv_input)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid CSV-formatted Pandas DataFrame produced using the"
                " `pandas.DataFrame.to_csv()` method."
            ),
            error_code=BAD_REQUEST,
        )


def parse_split_oriented_json_input_to_numpy(json_input):
    """
    :param json_input: A JSON-formatted string representation of a Pandas DataFrame with split
                       orient, or a stream containing such a string representation.
    """

    try:
        json_input_list = json.loads(json_input, object_pairs_hook=OrderedDict)
        return pd.DataFrame(
            index=json_input_list["index"],
            data=np.array(json_input_list["data"], dtype=object),
            columns=json_input_list["columns"],
        ).infer_objects()
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Numpy. Ensure that the input is"
                " a valid JSON-formatted Pandas DataFrame with the split orient"
                " produced using the `pandas.DataFrame.to_json(..., orient='split')`"
                " method."
            ),
            error_code=BAD_REQUEST,
        )


def predictions_to_json(raw_predictions, output):
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    wrapper_attr_name = os.environ.get(PREDICTIONS_WRAPPER_ATTR_NAME_ENV_KEY, None)
    if wrapper_attr_name:
        predictions = {wrapper_attr_name: predictions}
    json.dump(predictions, output, cls=NumpyEncoder)


def _handle_serving_error(error_message, error_code, include_traceback=True):
    """
    Logs information about an exception thrown by model inference code that is currently being
    handled and reraises it with the specified error message. The exception stack trace
    is also included in the reraised error message.

    :param error_message: A message for the reraised exception.
    :param error_code: An appropriate error code for the reraised exception. This should be one of
                       the codes listed in the `mlflow.protos.databricks_pb2` proto.
    :param include_traceback: Whether to include the current traceback in the returned error.
    """
    if include_traceback:
        traceback_buf = StringIO()
        traceback.print_exc(file=traceback_buf)
        traceback_str = traceback_buf.getvalue()
        e = MlflowException(message=error_message, error_code=error_code, stack_trace=traceback_str)
    else:
        e = MlflowException(message=error_message, error_code=error_code)
    reraise(MlflowException, e)


def init(model: PyFuncModel):

    """
    Initialize the server. Loads pyfunc model from the path.
    """
    app = flask.Flask(__name__)
    input_schema = model.metadata.get_input_schema()

    @app.route("/ping", methods=["GET"])
    def ping():  # pylint: disable=unused-variable
        """
        Determine if the container is working and healthy.
        We declare it healthy if we can load the model successfully.
        """
        health = model is not None
        status = 200 if health else 404
        return flask.Response(response="\n", status=status, mimetype="application/json")

    @app.route("/invocations", methods=["POST"])
    @catch_mlflow_exception
    def transformation():  # pylint: disable=unused-variable
        """
        Do an inference on a single batch of data. In this sample server,
        we take data as CSV or json, convert it to a Pandas DataFrame or Numpy,
        generate predictions and convert them back to json.
        """

        # Content-Type can include other attributes like CHARSET
        # Content-type RFC: https://datatracker.ietf.org/doc/html/rfc2045#section-5.1
        # TODO: Suport ";" in quoted parameter values
        type_parts = flask.request.content_type.split(";")
        type_parts = list(map(str.strip, type_parts))
        mime_type = type_parts[0]
        parameter_value_pairs = type_parts[1:]
        parameter_values = {}
        for parameter_value_pair in parameter_value_pairs:
            (key, _, value) = parameter_value_pair.partition("=")
            parameter_values[key] = value

        charset = parameter_values.get("charset", "utf-8").lower()
        if charset != "utf-8":
            return flask.Response(
                response="The scoring server only supports UTF-8",
                status=415,
                mimetype="text/plain",
            )

        content_format = parameter_values.get("format")

        # Convert from CSV to pandas
        if mime_type == CONTENT_TYPE_CSV and not content_format:
            data = flask.request.data.decode("utf-8")
            csv_input = StringIO(data)
            data = parse_csv_input(csv_input=csv_input)
        elif mime_type == CONTENT_TYPE_JSON and not content_format:
            json_str = flask.request.data.decode("utf-8")
            data = infer_and_parse_json_input(json_str, input_schema)
        elif (
            mime_type == CONTENT_TYPE_JSON and content_format == CONTENT_TYPE_FORMAT_SPLIT_ORIENTED
        ):
            data = parse_json_input(
                json_input=StringIO(flask.request.data.decode("utf-8")),
                orient="split",
                schema=input_schema,
            )
        elif (
            mime_type == CONTENT_TYPE_JSON
            and content_format == CONTENT_TYPE_FORMAT_RECORDS_ORIENTED
        ):
            data = parse_json_input(
                json_input=StringIO(flask.request.data.decode("utf-8")),
                orient="records",
                schema=input_schema,
            )
        elif mime_type == CONTENT_TYPE_JSON_SPLIT_NUMPY and not content_format:
            data = parse_split_oriented_json_input_to_numpy(flask.request.data.decode("utf-8"))
        else:
            return flask.Response(
                response=(
                    "This predictor only supports the following content types and formats:"
                    " Types: {supported_content_types}; Formats: {formats}."
                    " Got '{received_content_type}'.".format(
                        supported_content_types=CONTENT_TYPES,
                        formats=FORMATS,
                        received_content_type=flask.request.content_type,
                    )
                ),
                status=415,
                mimetype="text/plain",
            )

        # Do the prediction

        try:
            raw_predictions = model.predict(data)
        except MlflowException as e:
            _handle_serving_error(
                error_message=e.message, error_code=BAD_REQUEST, include_traceback=False
            )
        except Exception:
            _handle_serving_error(
                error_message=(
                    "Encountered an unexpected error while evaluating the model. Verify"
                    " that the serialized input Dataframe is compatible with the model for"
                    " inference."
                ),
                error_code=BAD_REQUEST,
            )
        result = StringIO()
        predictions_to_json(raw_predictions, result)
        return flask.Response(response=result.getvalue(), status=200, mimetype="application/json")

    return app


def _predict(model_uri, input_path, output_path, content_type, json_format):
    pyfunc_model = load_model(model_uri)
    if input_path is None:
        input_path = sys.stdin

    if content_type == "json":
        df = parse_json_input(input_path, orient=json_format)
    elif content_type == "csv":
        df = parse_csv_input(input_path)
    else:
        raise Exception("Unknown content type '{}'".format(content_type))

    if output_path is None:
        predictions_to_json(pyfunc_model.predict(df), sys.stdout)
    else:
        with open(output_path, "w") as fout:
            predictions_to_json(pyfunc_model.predict(df), fout)


def _serve(model_uri, port, host):
    pyfunc_model = load_model(model_uri)
    init(pyfunc_model).run(port=port, host=host)


def get_cmd(
    model_uri: str, port: int = None, host: int = None, nworkers: int = None
) -> Tuple[str, Dict[str, str]]:
    local_uri = path_to_local_file_uri(model_uri)
    # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
    # platform compatibility.
    if os.name != "nt":
        args = ["--timeout=60"]
        if port and host:
            args.append(f"-b {host}:{port}")
        elif host:
            args.append(f"-b {host}")

        if nworkers:
            args.append(f"-w {nworkers}")

        command = (
            f"gunicorn {' '.join(args)} ${{GUNICORN_CMD_ARGS}}"
            " -- mlflow.pyfunc.scoring_server.wsgi:app"
        )
    else:
        args = []
        if host:
            args.append(f"--host={host}")

        if port:
            args.append(f"--port={port}")

        command = (
            f"waitress-serve {' '.join(args)} "
            "--ident=mlflow mlflow.pyfunc.scoring_server.wsgi:app"
        )

    command_env = os.environ.copy()
    command_env[_SERVER_MODEL_PATH] = local_uri

    return command, command_env
