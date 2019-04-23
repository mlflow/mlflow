import os
import sys
import pandas as pd
import traceback

from six import reraise

from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST
from mlflow.exceptions import MlflowException


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def _add_code_to_system_path(code_path):
    sys.path = [code_path] + _get_code_dirs(code_path) + sys.path


def _get_code_dirs(src_code_path, dst_code_path=None):
    """
    Obtains the names of the subdirectories contained under the specified source code
    path and joins them with the specified destination code path.

    :param src_code_path: The path of the source code directory for which to list subdirectories.
    :param dst_code_path: The destination directory path to which subdirectory names should be
                          joined.
    """
    if not dst_code_path:
        dst_code_path = src_code_path
    return [(os.path.join(dst_code_path, x)) for x in os.listdir(src_code_path)
            if os.path.isdir(x) and not x == "__pycache__"]


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
        handle_serving_error(
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
        handle_serving_error(
                error_message=(
                    "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                    " a valid CSV-formatted Pandas DataFrame produced using the"
                    " `pandas.DataFrame.to_csv()` method."),
                error_code=MALFORMED_REQUEST)


def handle_serving_error(error_message, error_code):
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
