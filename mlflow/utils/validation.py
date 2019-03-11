"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import numbers
import os.path
import re

import numpy as np

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_VALID_PARAM_AND_METRIC_NAMES = re.compile(r"^[/\w.\- ]*$")

# Regex for valid run IDs: must be a 32-character hex string.
_RUN_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")

_BAD_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),"
    " spaces ( ), and slashes (/)."
)


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
        raise MlflowException("Invalid metric name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE),
                              INVALID_PARAMETER_VALUE)
    if path_not_unique(name):
        raise MlflowException("Invalid metric name: '%s'. %s" % (name, bad_path_message(name)),
                              INVALID_PARAMETER_VALUE)


def _validate_metric(key, value, timestamp):
    _validate_metric_name(key)
    if not isinstance(value, numbers.Number) or value > np.finfo(np.float64).max \
            or value < np.finfo(np.float64).min:
        raise MlflowException(
            "Got invalid value %s for metric '%s' (timestamp=%s). Please specify value as a valid "
            "double (64-bit floating point)" % (value, key, timestamp),
            INVALID_PARAMETER_VALUE)

    if not isinstance(timestamp, numbers.Number) or timestamp < 0 or \
            timestamp < np.iinfo(np.int64).min:
        raise MlflowException(
            "Got invalid timestamp %s for metric '%s' (value=%s). Timestamp must be a nonnegative "
            "long (64-bit integer) " % (timestamp, key, value),
            INVALID_PARAMETER_VALUE)


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException("Invalid parameter name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE),
                              INVALID_PARAMETER_VALUE)
    if path_not_unique(name):
        raise MlflowException("Invalid parameter name: '%s'. %s" % (name, bad_path_message(name)),
                              INVALID_PARAMETER_VALUE)


def _validate_tag_name(name):
    """Check that `name` is a valid tag name and raise an exception if it isn't."""
    # Reuse param & metric check.
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException("Invalid tag name: '%s'. %s" % (name, _BAD_CHARACTERS_MESSAGE),
                              INVALID_PARAMETER_VALUE)
    if path_not_unique(name):
        raise MlflowException("Invalid tag name: '%s'. %s" % (name, bad_path_message(name)),
                              INVALID_PARAMETER_VALUE)


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


def _validate_experiment_name(experiment_name):
    """Check that `experiment_name` is a valid string and raise an exception if it isn't."""
    if experiment_name == "" or experiment_name is None:
        raise MlflowException("Invalid experiment name: '%s'" % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
    if not isinstance(experiment_name, str):
        raise MlflowException("Invalid experiment name: %s. Expects a string." % experiment_name,
                              error_code=INVALID_PARAMETER_VALUE)
