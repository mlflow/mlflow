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
import logging


import flask

from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.utils.rest_utils import NumpyEncoder
from mlflow.server.handlers import catch_mlflow_exception
from mlflow.pyfunc.utils import (parse_csv_input, parse_json_input,
                                 handle_serving_error)

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from mlflow.utils import get_jsonable_obj

from mlflow.pyfunc.constants import (MODEL_ARTIFACT_PATH_VAR, CONTENT_TYPE_CSV,
                                     CONTENT_TYPE_JSON, CONTENT_TYPE_JSON_SPLIT_ORIENTED,
                                     CONTENT_TYPE_JSON_RECORDS_ORIENTED, CONTENT_TYPES)


_logger = logging.getLogger(__name__)


def _load_model():
    import os
    from mlflow.pyfunc import load_pyfunc
    model_path = os.environ.get(MODEL_ARTIFACT_PATH_VAR)
    model_obj = None
    if model_path:
        model_obj = load_pyfunc(model_path)
    return model_obj


app = flask.Flask(__name__)
model = _load_model()


@app.route('/ping', methods=['GET'])
def ping():  # pylint: disable=unused-variable
    """
    Determine if the container is working and healthy.
    We declare it healthy if we can load the model successfully.
    """
    health = model is not None
    status = 200 if health else 404
    response = 'Model Loaded' if health else 'Model not found'
    return flask.Response(response=response, status=status, mimetype='application/json')


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
        handle_serving_error(
                error_message=(
                    "Encountered an unexpected error while evaluating the model. Verify"
                    " that the serialized input Dataframe is compatible with the model for"
                    " inference."),
                error_code=BAD_REQUEST)

    predictions = get_jsonable_obj(raw_predictions, pandas_orient="records")
    result = json.dumps(predictions, cls=NumpyEncoder)
    return flask.Response(response=result, status=200, mimetype='application/json')
