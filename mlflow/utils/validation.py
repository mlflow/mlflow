"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import re

# Regex for metric and parameter names: we only allow [a-zA-Z0-9_-] ([\w-]), / and . characters,
# and we do not allow adjacent, leading or trailing / or path components that are empty or only
# contain dots. The goal is to make our names be valid, unique path names too in order to treat
# them as such in the UI and file store.
_METRIC_AND_PARAM_NAME_REGEX = re.compile(r"^([\w.-]*[\w-][\w.-]*)(/[\w.-]*[\w-][\w.-]*)*$")

# Regex for valid run IDs: must be a 32-character hex string.
_RUN_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")


def _validate_metric_name(name):
    """Check that `name` is a valid metric name and raise an exception if it isn't."""
    if _METRIC_AND_PARAM_NAME_REGEX.match(name) is None:
        raise Exception("Invalid metric name: '%s'" % name)


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if _METRIC_AND_PARAM_NAME_REGEX.match(name) is None:
        raise Exception("Invalid parameter name: '%s'" % name)


def _validate_run_id(run_id):
    """Check that `run_id` is a valid run ID and raise an exception if it isn't."""
    if _RUN_ID_REGEX.match(run_id) is None:
        raise Exception("Invalid run ID: '%s'" % run_id)
