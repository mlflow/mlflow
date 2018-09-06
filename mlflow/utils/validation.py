"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import os.path
import re

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
        raise Exception("Invalid run ID: '%s'" % run_id)
