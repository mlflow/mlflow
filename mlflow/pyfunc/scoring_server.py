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

import sys
import json
import traceback

import pandas as pd
import flask
from six import reraise

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST 
from mlflow.utils.rest_utils import NumpyEncoder
from mlflow.server.handlers import catch_mlflow_exception 

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from mlflow.utils import get_jsonable_obj


def parse_json_input(json_input):
    """
    :param json_input: A JSON-formatted string representation of a Pandas Dataframe, or a stream 
                       containing such a string representation.
    """
    try:
        return pd.read_json(json_input, orient="split")
    except Exception as e:
        _handle_input_parsing_error(reraised_error_text=
                ("Failed to parse input as a Pandas Dataframe. Please ensure that the input is"
                 " a valid JSON-formatted Pandas Dataframe with the `split` orientation produced"
                 " using the `pandas.DataFrame.to_json(..., orient='split')` method. Original"
                 " exception text: {exception_text}".format(exception_text=str(e))))


def parse_csv_input(csv_input):
    """
    :param csv_input: A CSV-formatted string representation of a Pandas Dataframe, or a stream
                      containing such a string representation.
    """
    try:
        return pd.read_csv(csv_input)
    except Exception as e:
        _handle_input_parsing_error(reraised_error_text=
                ("Failed to parse input as a Pandas Dataframe. Please ensure that the input is"
                 " a valid CSV-formatted Pandas Dataframe produced using the" 
                 " `pandas.DataFrame.to_csv()` method. Original exception text:" 
                 " {exception_text}".format(exception_text=str(e))))


def _handle_input_parsing_error(reraised_error_text):
    traceback.print_exc()
    tb = sys.exc_info()[2]
    reraise(MlflowException,
            MlflowException(message=reraised_error_text, error_code=MALFORMED_REQUEST),
            tb)


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
        we take data as CSV or json, convert it to a pandas data frame,
        generate predictions and convert them back to CSV.
        """
        # Convert from CSV to pandas
        if flask.request.content_type == 'text/csv':
            data = flask.request.data.decode('utf-8')
            csv_input = StringIO(data)
            data = parse_csv_input(csv_input=csv_input)
        elif flask.request.content_type == 'application/json':
            data = flask.request.data.decode('utf-8')
            json_input = StringIO(data)
            data = parse_json_input(json_input=json_input)
        else:
            return flask.Response(
                response='This predictor only supports CSV or JSON  data, got %s' % str(
                    flask.request.content_type), status=415, mimetype='text/plain')

        # Do the prediction
        predictions = get_jsonable_obj(model.predict(data))
        result = json.dumps(predictions, cls=NumpyEncoder)
        return flask.Response(response=result, status=200, mimetype='application/json')

    return app
