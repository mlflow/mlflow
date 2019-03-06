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
MAX_PARAM_LENGTH = 500
MAX_TAG_LENGTH = (1 << 16) - 1
MAX_ENTITY_KEY_LENGTH = 250


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
    _validate_length_limit("Metric name", MAX_ENTITY_KEY_LENGTH, name)


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise Exception("Invalid parameter name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE))
    if path_not_unique(name):
        raise Exception("Invalid parameter name: '%s'. %s" % (name, bad_path_message(name)))
    _validate_length_limit("Param name", MAX_ENTITY_KEY_LENGTH, name)


def _validate_tag_name(name):
    """Check that `name` is a valid tag name and raise an exception if it isn't."""
    # Reuse param & metric check.
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise Exception("Invalid tag name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE))
    if path_not_unique(name):
        raise Exception("Invalid tag name: '%s'. %s" % (name, bad_path_message(name)))
    _validate_length_limit("Tag name", MAX_ENTITY_KEY_LENGTH, name)


def _validate_length_limit(entity_name, limit, value):
    if len(value) > limit:
        raise MlflowException(
            "%s %s had length %s, which exceeded length limit of %s" %
            (entity_name, value, len(value), limit))


def _validate_param_value(value):
    _validate_length_limit("Param value", MAX_PARAM_LENGTH, value)


def _validate_tag_value(value):
    _validate_length_limit("Tag value", MAX_PARAM_LENGTH, value)


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
    _validate_batch_limit(entity_name="params", limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(params))
    _validate_batch_limit(entity_name="tags", limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(tags))
    total_length = len(metrics) + len(params) + len(tags)
    _validate_batch_limit(entity_name="metrics, params, and tags",
                          limit=MAX_ENTITIES_PER_BATCH, length=total_length)


def _validate_batch_log_data(metrics, params, tags):
    for metric in metrics:
        _validate_metric_name(metric.key)
    for param in params:
        _validate_param_name(param.key)
        _validate_param_value(param.value)
    for tag in tags:
        _validate_tag_name(tag.key)
        _validate_tag_value(tag.value)
    # Verify upfront that the user isn't attempting to overwrite any params within their
    # batched logging request
    param_map = {}
    for param in params:
        if param.key not in param_map:
            param_map[param.key] = param.value
        elif param.key in param_map and param_map[param.key] != param.value:
            raise MlflowException("Param %s had existing value %s, refusing to overwrite with "
                                  "new value %s. Please log the new param value under a different "
                                  "param name." % (param.key, param_map[param.key], param.value))


def _validate_experiment_name(experiment_name):
    """Check that `experiment_name` is a valid string and raise an exception if it isn't."""
    if experiment_name == "" or experiment_name is None:
        raise MlflowException("Invalid experiment name: '%s'" % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(experiment_name, str):
        raise MlflowException("Invalid experiment name: %s. Expects a string." % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
