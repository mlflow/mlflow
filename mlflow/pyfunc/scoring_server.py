"""
Scoring server for python model format.
The passed int model is expected to have function:
   predict(pandas.Dataframe) -> pandas.DataFrame

Input, expected intext/csv or application/json format,
is parsed into pandas.DataFrame and passed to the model.

Defines two endpoints:
    /ping used for health check
    /invocations used for scoring
"""
from __future__ import print_function

import flask
import json
from json import JSONEncoder
import logging
import numpy as np
import pandas as pd
from six import reraise
import subprocess
import sys
import traceback

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the verison of
# mlflow int the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException

try:
    from mlflow.pyfunc import load_model
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model
from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST, BAD_REQUEST
from mlflow.server.handlers import catch_mlflow_exception
from mlflow.projects import _get_or_create_conda_env, _get_conda_bin_executable

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_JSON_RECORDS_ORIENTED = "application/json; format=pandas-records"
CONTENT_TYPE_JSON_SPLIT_ORIENTED = "application/json; format=pandas-split"

CONTENT_TYPES = [
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_ORIENTED
]

_logger = logging.getLogger(__name__)


def parse_json_input(json_input, orient="split"):
    """
    :param json_input: A JSON-formatted string representation of a Pandas DataFrame, or a stream
                       containing such a string representation.
    :param orient: The Pandas DataFrame orientation of the JSON input. This is either 'split'
                   or 'records'.
    """
    # pylint: disable=broad-except
    try:
        return pd.read_json(json_input, orient=orient, dtype=False)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid JSON-formatted Pandas DataFrame with the `{orient}` orient"
                " produced using the `pandas.DataFrame.to_json(..., orient='{orient}')`"
                " method.".format(orient=orient)),
            error_code=MALFORMED_REQUEST)


def parse_csv_input(csv_input):
    """
    :param csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                      containing such a string representation.
    """
    # pylint: disable=broad-except
    try:
        return pd.read_csv(csv_input)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid CSV-formatted Pandas DataFrame produced using the"
                " `pandas.DataFrame.to_csv()` method."),
            error_code=MALFORMED_REQUEST)


def predictions_to_json(raw_predictions, output):
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    json.dump(predictions, output, cls=NumpyEncoder)


def _handle_serving_error(error_message, error_code):
    """
    Logs information about an exception thrown by model inference code that is currently being
    handled and reraises it with the specified error message. The exception stack trace
    is also included in the reraised error message.

    :param error_message: A message for the reraised exception.
    :param error_code: An appropriate error code for the reraised exception. This should be one of
                       the codes listed in the `mlflow.protos.databricks_pb2` proto.
    """
    traceback_buf = StringIO()
    traceback.print_exc(file=traceback_buf)
    reraise(MlflowException,
            MlflowException(
                message=error_message,
                error_code=error_code,
                stack_trace=traceback_buf.getvalue()))


def init(model):
    """
    Initialize the server. Loads pyfunc model from the path.
    """
    app = flask.Flask(__name__)

    @app.route('/ping', methods=['GET'])
    def ping():  # pylint: disable=unused-variable
        """
        Determine if the container is working and healthy.
        We declare it healthy if we can load the model successfully.
        """
        health = model is not None
        status = 200 if health else 404
        return flask.Response(response='\n', status=status, mimetype='application/json')

    @app.route('/invocations', methods=['POST'])
    @catch_mlflow_exception
    def transformation():  # pylint: disable=unused-variable
        """
        Do an inference on a single batch of data. In this sample server,
        we take data as CSV or json, convert it to a Pandas DataFrame,
        generate predictions and convert them back to CSV.
        """
        # Convert from CSV to pandas
        if flask.request.content_type == CONTENT_TYPE_CSV:
            data = flask.request.data.decode('utf-8')
            csv_input = StringIO(data)
            data = parse_csv_input(csv_input=csv_input)
        elif flask.request.content_type in [CONTENT_TYPE_JSON, CONTENT_TYPE_JSON_SPLIT_ORIENTED]:
            data = parse_json_input(json_input=flask.request.data.decode('utf-8'),
                                    orient="split")
        elif flask.request.content_type == CONTENT_TYPE_JSON_RECORDS_ORIENTED:
            data = parse_json_input(json_input=flask.request.data.decode('utf-8'),
                                    orient="records")
        else:
            return flask.Response(
                response=("This predictor only supports the following content types,"
                          " {supported_content_types}. Got '{received_content_type}'.".format(
                            supported_content_types=CONTENT_TYPES,
                            received_content_type=flask.request.content_type)),
                status=415,
                mimetype='text/plain')

        # Do the prediction
        # pylint: disable=broad-except
        try:
            raw_predictions = model.predict(data)
        except Exception:
            _handle_serving_error(
                error_message=(
                    "Encountered an unexpected error while evaluating the model. Verify"
                    " that the serialized input Dataframe is compatible with the model for"
                    " inference."),
                error_code=BAD_REQUEST)
        result = StringIO()
        predictions_to_json(raw_predictions, result)
        return flask.Response(response=result.getvalue(), status=200, mimetype='application/json')

    return app


def _safe_local_path(local_path):
    # NB: Since mlflow 1.0, mlflow.pyfunc.load_model expects uri instead of local path. Local paths
    # work, however absolute windows path don't because the drive is parsed as scheme. Since we do
    # not control the version of mlflow (the scoring server version matches the version of mlflow
    # invoking the scoring command, the rest of mlflow comes from the model environment) we check
    # the mlflow version at run time and convert local path to file uri to ensure platform
    # independence.
    from mlflow.version import VERSION
    is_recent_version = VERSION.endswith("dev0") or int(VERSION.split(".")[0]) >= 1
    if is_recent_version:
        from mlflow.tracking.utils import path_to_local_file_uri
        return path_to_local_file_uri(local_path)
    return local_path


def _predict(local_path, input_path, output_path, content_type, json_format):
    pyfunc_model = load_model(_safe_local_path(local_path))
    if input_path is None or input_path == "__stdin__":
        input_path = sys.stdin

    if content_type == "json":
        df = parse_json_input(input_path, orient=json_format)
    elif content_type == "csv":
        df = parse_csv_input(input_path)
    else:
        raise Exception("Unknown content type '{}'".format(content_type))

    if output_path is None or output_path == "__stdout__":
        predictions_to_json(pyfunc_model.predict(df), sys.stdout)
    else:
        with open(output_path, "w") as fout:
            predictions_to_json(pyfunc_model.predict(df), fout)


def _serve(local_path, port, host):
    pyfunc_model = load_model(_safe_local_path(local_path))
    init(pyfunc_model).run(port=port, host=host)


def _execute_in_conda_env(conda_env_path, command):
    conda_env_name = _get_or_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    command = " && ".join(
        ["source {} {}".format(activate_path, conda_env_name), "pip install mlflow 1>&2", command]
    )
    _logger.info("=== Running command '%s'", command)
    child = subprocess.Popen(["bash", "-c", command], close_fds=True)
    rc = child.wait()
    if rc != 0:
        raise Exception("Command '{0}' returned non zero return code. Return code = {1}".format(
            command, rc
        ))


class NumpyEncoder(JSONEncoder):
    """ Special json encoder for numpy types.
    Note that some numpy types doesn't have native python equivalence,
    hence json.dumps will raise TypeError.
    In this case, you'll need to convert your numpy types into its closest python equivalence.
    """

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, np.generic):
            return np.asscalar(o)
        return JSONEncoder.default(self, o)


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


if __name__ == '__main__':
    if sys.argv[1] == "predict":
        _predict(local_path=sys.argv[2], input_path=sys.argv[3], output_path=sys.argv[4],
                 content_type=sys.argv[5], json_format=sys.argv[6])
    elif sys.argv[1] == "serve":
        _serve(local_path=sys.argv[2], port=sys.argv[3], host=sys.argv[4])
    else:
        raise Exception("Unknown command '{}'".format(sys.argv[1]))
