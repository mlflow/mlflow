"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import os.path
import re

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_VALID_PARAM_AND_METRIC_NAMES = re.compile(r"^[/\w.\- ]*$")

# Regex for valid run IDs: must be a 32-character hex string.
_RUN_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")

_BAD_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),"
    " spaces ( ), and slashes (/)."
)

MAX_PARAMS_TAGS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
MAX_ENTITIES_PER_BATCH = 1000
MAX_BATCH_LOG_REQUEST_SIZE = int(1e7)

def bad_path_message(name):
    return (
        "Names may be treated as files in certain cases, and must not resolve to other names"
        " when treated as such. This name would resolve to '%s'"
    ) % os.path.normpath(name)


def path_not_unique(name):
    norm = os.path.normpath(name)
    return norm != name or norm == '.' or norm.startswith('..') or norm.startswith('/')


def _validate_metric_name(name):
    """Check that `name` is a valid metric name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise Exception("Invalid metric name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE))
    if path_not_unique(name):
        raise Exception("Invalid metric name: '%s'. %s" % (name, bad_path_message(name)))


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise Exception("Invalid parameter name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE))
    if path_not_unique(name):
        raise Exception("Invalid parameter name: '%s'. %s" % (name, bad_path_message(name)))


def _validate_tag_name(name):
    """Check that `name` is a valid tag name and raise an exception if it isn't."""
    # Reuse param & metric check.
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise Exception("Invalid tag name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE))
    if path_not_unique(name):
        raise Exception("Invalid tag name: '%s'. %s" % (name, bad_path_message(name)))


def _validate_run_id(run_id):
    """Check that `run_id` is a valid run ID and raise an exception if it isn't."""
    if _RUN_ID_REGEX.match(run_id) is None:
        raise MlflowException("Invalid run ID: '%s'" % run_id, error_code=INVALID_PARAMETER_VALUE)


def _validate_experiment_id(exp_id):
    """Check that `experiment_id`is a valid integer and raise an exception if it isn't."""
    try:
        int(exp_id)
    except ValueError:
        raise MlflowException("Invalid experiment ID: '%s'" % exp_id,
                              error_code=INVALID_PARAMETER_VALUE)


def _validate_batch_limit(entity_name, limit, length):
    if length > limit:
        error_msg = ("A batch logging request can contain at most {limit} {name}. "
                     "Got {count} {name}. Please split up {name} across multiple requests and try "
                     "again.").format(name=entity_name, count=length, limit=limit)
        raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)


def _validate_batch_log_limits(metrics, params, tags):
    """Validate that the provided batched logging arguments are within expected limits."""
    _validate_batch_limit(entity_name="metrics", limit=MAX_METRICS_PER_BATCH, length=len(metrics))
    _validate_batch_limit(entity_name="params", limit=MAX_METRICS_PER_BATCH, length=len(params))
    _validate_batch_limit(entity_name="tags", limit=MAX_METRICS_PER_BATCH, length=len(tags))
    total_length = len(metrics) + len(params) + len(tags)
    _validate_batch_limit(entity_name="metrics, params, and tags",
                          limit=MAX_ENTITIES_PER_BATCH, length=total_length)

def _validate_batch_log_api_req(json_req):
    if len(json_req) > MAX_BATCH_LOG_REQUEST_SIZE:
        error_msg = ("Batched logging API requests must be at most {limit} bytes, got "
                     "request of size {size}.").format(
            limit=MAX_BATCH_LOG_REQUEST_SIZE, size=len(json_req))
        raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)

def _validate_experiment_name(experiment_name):
    """Check that `experiment_name` is a valid string and raise an exception if it isn't."""
    if experiment_name == "" or experiment_name is None:
        raise MlflowException("Invalid experiment name: '%s'" % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(experiment_name, str):
        raise MlflowException("Invalid experiment name: %s. Expects a string." % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
