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

import json
import traceback
import logging

import pandas as pd
import flask
from six import reraise

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST, BAD_REQUEST
from mlflow.utils.rest_utils import NumpyEncoder
from mlflow.server.handlers import catch_mlflow_exception

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from mlflow.utils import get_jsonable_obj

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

        predictions = get_jsonable_obj(raw_predictions, pandas_orient="records")
        result = json.dumps(predictions, cls=NumpyEncoder)
        return flask.Response(response=result, status=200, mimetype='application/json')

    return app
