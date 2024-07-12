"""
Utilities for validating user inputs such as metric names and parameter names.
"""
import logging
import numbers
import posixpath
import re
from typing import List

from mlflow.entities import Dataset, DatasetInput, InputTag, Param, RunTag
from mlflow.environment_variables import MLFLOW_TRUNCATE_LONG_VALUES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.string_utils import is_string_type

_logger = logging.getLogger(__name__)

# Regex for valid param and metric names: may only contain slashes, alphanumerics,
# underscores, periods, dashes, and spaces.
_VALID_PARAM_AND_METRIC_NAMES = re.compile(r"^[/\w.\- ]*$")

# Regex for valid run IDs: must be an alphanumeric string of length 1 to 256.
_RUN_ID_REGEX = re.compile(r"^[a-zA-Z0-9][\w\-]{0,255}$")

# Regex: starting with an alphanumeric, optionally followed by up to 63 characters
# including alphanumerics, underscores, or dashes.
_EXPERIMENT_ID_REGEX = re.compile(r"^[a-zA-Z0-9][\w\-]{0,63}$")

# Regex for valid registered model alias names: may only contain alphanumerics,
# underscores, and dashes.
_REGISTERED_MODEL_ALIAS_REGEX = re.compile(r"^[\w\-]*$")

# Regex for valid registered model alias to prevent conflict with version aliases.
_REGISTERED_MODEL_ALIAS_VERSION_REGEX = re.compile(r"^[vV]\d+$")

_BAD_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),"
    " spaces ( ), and slashes (/)."
)

_BAD_ALIAS_CHARACTERS_MESSAGE = (
    "Names may only contain alphanumerics, underscores (_), and dashes (-)."
)

_MISSING_KEY_NAME_MESSAGE = "A key name must be provided."

MAX_PARAMS_TAGS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
MAX_DATASETS_PER_BATCH = 1000
MAX_ENTITIES_PER_BATCH = 1000
MAX_BATCH_LOG_REQUEST_SIZE = int(1e6)
MAX_PARAM_VAL_LENGTH = 6000
MAX_TAG_VAL_LENGTH = 5000
MAX_EXPERIMENT_NAME_LENGTH = 500
MAX_EXPERIMENT_TAG_KEY_LENGTH = 250
MAX_EXPERIMENT_TAG_VAL_LENGTH = 5000
MAX_ENTITY_KEY_LENGTH = 250
MAX_MODEL_REGISTRY_TAG_KEY_LENGTH = 250
MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH = 5000
MAX_EXPERIMENTS_LISTED_PER_PAGE = 50000
MAX_DATASET_NAME_SIZE = 500
MAX_DATASET_DIGEST_SIZE = 36
# 1MB -1, the db limit for MEDIUMTEXT column is 16MB, but
# we restrict to 1MB because user might log lots of datasets
# to a single run, 16MB increases burden on db
MAX_DATASET_SCHEMA_SIZE = 1048575
MAX_DATASET_SOURCE_SIZE = 65535  # 64KB -1 (the db limit for TEXT column)
MAX_DATASET_PROFILE_SIZE = 16777215  # 16MB -1 (the db limit for MEDIUMTEXT column)
MAX_INPUT_TAG_KEY_SIZE = 255
MAX_INPUT_TAG_VALUE_SIZE = 500
MAX_REGISTERED_MODEL_ALIAS_LENGTH = 255
MAX_TRACE_TAG_KEY_LENGTH = 250
MAX_TRACE_TAG_VAL_LENGTH = 8000

_UNSUPPORTED_DB_TYPE_MSG = "Supported database engines are {%s}" % ", ".join(DATABASE_ENGINES)

PARAM_VALIDATION_MSG = """

The cause of this error is typically due to repeated calls
to an individual run_id event logging.

Incorrect Example:
---------------------------------------
with mlflow.start_run():
    mlflow.log_param("depth", 3)
    mlflow.log_param("depth", 5)
---------------------------------------

Which will throw an MlflowException for overwriting a
logged parameter.

Correct Example:
---------------------------------------
with mlflow.start_run():
    with mlflow.start_run(nested=True):
        mlflow.log_param("depth", 3)
    with mlflow.start_run(nested=True):
        mlflow.log_param("depth", 5)
---------------------------------------

Which will create a new nested run for each individual
model and prevent parameter key collisions within the
tracking store."""


def bad_path_message(name):
    return (
        "Names may be treated as files in certain cases, and must not resolve to other names"
        " when treated as such. This name would resolve to '%s'"
    ) % posixpath.normpath(name)


def path_not_unique(name):
    norm = posixpath.normpath(name)
    return norm != name or norm == "." or norm.startswith("..") or norm.startswith("/")


def _validate_metric_name(name):
    """Check that `name` is a valid metric name and raise an exception if it isn't."""
    if name is None:
        raise MlflowException(
            f"Metric name cannot be None. {_MISSING_KEY_NAME_MESSAGE}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException(
            f"Invalid metric name: '{name}'. {_BAD_CHARACTERS_MESSAGE}",
            INVALID_PARAMETER_VALUE,
        )
    if path_not_unique(name):
        raise MlflowException(
            f"Invalid metric name: '{name}'. {bad_path_message(name)}",
            INVALID_PARAMETER_VALUE,
        )


def _is_numeric(value):
    """
    Returns True if the passed-in value is numeric.
    """
    # Note that `isinstance(bool_value, numbers.Number)` returns `True` because `bool` is a
    # subclass of `int`.
    return not isinstance(value, bool) and isinstance(value, numbers.Number)


def _validate_metric(key, value, timestamp, step):
    """
    Check that a metric with the specified key, value, timestamp, and step is valid and raise an
    exception if it isn't.
    """
    _validate_metric_name(key)
    # value must be a Number
    # since bool is an instance of Number check for bool additionally
    if not _is_numeric(value):
        raise MlflowException(
            f"Got invalid value {value} for metric '{key}' (timestamp={timestamp}). "
            "Please specify value as a valid double (64-bit floating point)",
            INVALID_PARAMETER_VALUE,
        )

    if not isinstance(timestamp, numbers.Number) or timestamp < 0:
        raise MlflowException(
            f"Got invalid timestamp {timestamp} for metric '{key}' (value={value}). "
            "Timestamp must be a nonnegative long (64-bit integer) ",
            INVALID_PARAMETER_VALUE,
        )

    if not isinstance(step, numbers.Number):
        raise MlflowException(
            f"Got invalid step {step} for metric '{key}' (value={value}). "
            "Step must be a valid long (64-bit integer).",
            INVALID_PARAMETER_VALUE,
        )

    _validate_length_limit("Metric name", MAX_ENTITY_KEY_LENGTH, key)


def _validate_param(key, value):
    """
    Check that a param with the specified key & value is valid and raise an exception if it
    isn't.
    """
    _validate_param_name(key)
    return Param(
        _validate_length_limit("Param key", MAX_ENTITY_KEY_LENGTH, key),
        _validate_length_limit("Param value", MAX_PARAM_VAL_LENGTH, value, truncate=True),
    )


def _validate_tag(key, value):
    """
    Check that a tag with the specified key & value is valid and raise an exception if it isn't.
    """
    _validate_tag_name(key)
    return RunTag(
        _validate_length_limit("Tag key", MAX_ENTITY_KEY_LENGTH, key),
        _validate_length_limit("Tag value", MAX_TAG_VAL_LENGTH, value, truncate=True),
    )


def _validate_experiment_tag(key, value):
    """
    Check that a tag with the specified key & value is valid and raise an exception if it isn't.
    """
    _validate_tag_name(key)
    _validate_length_limit("Tag key", MAX_EXPERIMENT_TAG_KEY_LENGTH, key)
    _validate_length_limit("Tag value", MAX_EXPERIMENT_TAG_VAL_LENGTH, value)


def _validate_registered_model_tag(key, value):
    """
    Check that a tag with the specified key & value is valid and raise an exception if it isn't.
    """
    _validate_tag_name(key)
    _validate_length_limit("Registered model key", MAX_MODEL_REGISTRY_TAG_KEY_LENGTH, key)
    _validate_length_limit("Registered model value", MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH, value)


def _validate_model_version_tag(key, value):
    """
    Check that a tag with the specified key & value is valid and raise an exception if it isn't.
    """
    _validate_tag_name(key)
    _validate_tag_value(value)
    _validate_length_limit("Model version key", MAX_MODEL_REGISTRY_TAG_KEY_LENGTH, key)
    _validate_length_limit("Model version value", MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH, value)


def _validate_param_keys_unique(params):
    """Ensures that duplicate param keys are not present in the `log_batch()` params argument"""
    unique_keys = []
    dupe_keys = []
    for param in params:
        if param.key not in unique_keys:
            unique_keys.append(param.key)
        else:
            dupe_keys.append(param.key)

    if dupe_keys:
        raise MlflowException(
            f"Duplicate parameter keys have been submitted: {dupe_keys}. Please ensure "
            "the request contains only one param value per param key.",
            INVALID_PARAMETER_VALUE,
        )


def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if name is None:
        raise MlflowException(
            f"Parameter name cannot be None. {_MISSING_KEY_NAME_MESSAGE}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException(
            f"Invalid parameter name: '{name}'. {_BAD_CHARACTERS_MESSAGE}",
            INVALID_PARAMETER_VALUE,
        )
    if path_not_unique(name):
        raise MlflowException(
            f"Invalid parameter name: '{name}'. {bad_path_message(name)}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_tag_name(name):
    """Check that `name` is a valid tag name and raise an exception if it isn't."""
    # Reuse param & metric check.
    if name is None:
        raise MlflowException(
            f"Tag name cannot be None. {_MISSING_KEY_NAME_MESSAGE}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException(
            f"Invalid tag name: '{name}'. {_BAD_CHARACTERS_MESSAGE}",
            INVALID_PARAMETER_VALUE,
        )
    if path_not_unique(name):
        raise MlflowException(
            f"Invalid tag name: '{name}'. {bad_path_message(name)}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_length_limit(entity_name, limit, value, *, truncate=False):
    if value is None:
        return None

    if len(value) <= limit:
        return value

    if truncate and MLFLOW_TRUNCATE_LONG_VALUES.get():
        _logger.warning(
            f"{entity_name} '{value[:100]}...' ({len(value)} characters) is truncated to "
            f"{limit} characters to meet the length limit."
        )
        return value[:limit]

    raise MlflowException(
        f"{entity_name} '{value[:250]}' had length {len(value)}, "
        f"which exceeded length limit of {limit}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _validate_run_id(run_id):
    """Check that `run_id` is a valid run ID and raise an exception if it isn't."""
    if _RUN_ID_REGEX.match(run_id) is None:
        raise MlflowException(f"Invalid run ID: '{run_id}'", error_code=INVALID_PARAMETER_VALUE)


def _validate_experiment_id(exp_id):
    """Check that `experiment_id`is a valid string or None, raise an exception if it isn't."""
    if exp_id is not None and _EXPERIMENT_ID_REGEX.match(exp_id) is None:
        raise MlflowException(
            f"Invalid experiment ID: '{exp_id}'", error_code=INVALID_PARAMETER_VALUE
        )


def _validate_batch_limit(entity_name, limit, length):
    if length > limit:
        error_msg = (
            f"A batch logging request can contain at most {limit} {entity_name}. "
            f"Got {length} {entity_name}. Please split up {entity_name} across multiple"
            " requests and try again."
        )
        raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)


def _validate_batch_log_limits(metrics, params, tags):
    """Validate that the provided batched logging arguments are within expected limits."""
    _validate_batch_limit(entity_name="metrics", limit=MAX_METRICS_PER_BATCH, length=len(metrics))
    _validate_batch_limit(entity_name="params", limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(params))
    _validate_batch_limit(entity_name="tags", limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(tags))
    total_length = len(metrics) + len(params) + len(tags)
    _validate_batch_limit(
        entity_name="metrics, params, and tags", limit=MAX_ENTITIES_PER_BATCH, length=total_length
    )


def _validate_batch_log_data(metrics, params, tags):
    for metric in metrics:
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
    return (
        metrics,
        [_validate_param(p.key, p.value) for p in params],
        [_validate_tag(t.key, t.value) for t in tags],
    )


def _validate_batch_log_api_req(json_req):
    if len(json_req) > MAX_BATCH_LOG_REQUEST_SIZE:
        error_msg = (
            "Batched logging API requests must be at most {limit} bytes, got a "
            "request of size {size}."
        ).format(limit=MAX_BATCH_LOG_REQUEST_SIZE, size=len(json_req))
        raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)


def _validate_experiment_name(experiment_name):
    """Check that `experiment_name` is a valid string and raise an exception if it isn't."""
    if experiment_name == "" or experiment_name is None:
        raise MlflowException(
            f"Invalid experiment name: '{experiment_name}'", error_code=INVALID_PARAMETER_VALUE
        )

    if not is_string_type(experiment_name):
        raise MlflowException(
            f"Invalid experiment name: {experiment_name}. Expects a string.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if len(experiment_name) > MAX_EXPERIMENT_NAME_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"Experiment name exceeds the maximum length of {MAX_EXPERIMENT_NAME_LENGTH} characters"
        )


def _validate_experiment_id_type(experiment_id):
    """
    Check that a user-provided experiment_id is either a string, int, or None and raise an
    exception if it isn't.
    """
    if experiment_id is not None and not isinstance(experiment_id, (str, int)):
        raise MlflowException(
            f"Invalid experiment id: {experiment_id} of type {type(experiment_id)}. "
            "Must be one of str, int, or None.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_model_name(model_name):
    if model_name is None or model_name == "":
        raise MlflowException("Registered model name cannot be empty.", INVALID_PARAMETER_VALUE)


def _validate_model_version(model_version):
    try:
        model_version = int(model_version)
    except ValueError:
        raise MlflowException(
            f"Model version must be an integer, got '{model_version}'",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_model_alias_name(model_alias_name):
    if model_alias_name is None or model_alias_name == "":
        raise MlflowException(
            "Registered model alias name cannot be empty.", INVALID_PARAMETER_VALUE
        )
    if not _REGISTERED_MODEL_ALIAS_REGEX.match(model_alias_name):
        raise MlflowException(
            f"Invalid alias name: '{model_alias_name}'. {_BAD_ALIAS_CHARACTERS_MESSAGE}",
            INVALID_PARAMETER_VALUE,
        )
    _validate_length_limit(
        "Registered model alias name", MAX_REGISTERED_MODEL_ALIAS_LENGTH, model_alias_name
    )
    if model_alias_name.lower() == "latest":
        raise MlflowException(
            "'latest' alias name (case insensitive) is reserved.", INVALID_PARAMETER_VALUE
        )
    if _REGISTERED_MODEL_ALIAS_VERSION_REGEX.match(model_alias_name):
        raise MlflowException(
            f"Version alias name '{model_alias_name}' is reserved.", INVALID_PARAMETER_VALUE
        )


def _validate_experiment_artifact_location(artifact_location):
    if artifact_location is not None and artifact_location.startswith("runs:"):
        raise MlflowException(
            f"Artifact location cannot be a runs:/ URI. Given: '{artifact_location}'",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_db_type_string(db_type):
    """validates db_type parsed from DB URI is supported"""
    if db_type not in DATABASE_ENGINES:
        error_msg = f"Invalid database engine: '{db_type}'. '{_UNSUPPORTED_DB_TYPE_MSG}'"
        raise MlflowException(error_msg, INVALID_PARAMETER_VALUE)


def _validate_model_version_or_stage_exists(version, stage):
    if version and stage:
        raise MlflowException("version and stage cannot be set together", INVALID_PARAMETER_VALUE)

    if not (version or stage):
        raise MlflowException("version or stage must be set", INVALID_PARAMETER_VALUE)


def _validate_tag_value(value):
    if value is None:
        raise MlflowException("Tag value cannot be None", INVALID_PARAMETER_VALUE)


def _validate_dataset_inputs(dataset_inputs: List[DatasetInput]):
    for dataset_input in dataset_inputs:
        _validate_dataset(dataset_input.dataset)
        _validate_input_tags(dataset_input.tags)


def _validate_dataset(dataset: Dataset):
    if dataset is None:
        raise MlflowException("Dataset cannot be None", INVALID_PARAMETER_VALUE)
    if dataset.name is None:
        raise MlflowException("Dataset name cannot be None", INVALID_PARAMETER_VALUE)
    if dataset.digest is None:
        raise MlflowException("Dataset digest cannot be None", INVALID_PARAMETER_VALUE)
    if dataset.source_type is None:
        raise MlflowException("Dataset source_type cannot be None", INVALID_PARAMETER_VALUE)
    if dataset.source is None:
        raise MlflowException("Dataset source cannot be None", INVALID_PARAMETER_VALUE)
    if len(dataset.name) > MAX_DATASET_NAME_SIZE:
        raise MlflowException(
            f"Dataset name exceeds the maximum length of {MAX_DATASET_NAME_SIZE}",
            INVALID_PARAMETER_VALUE,
        )
    if len(dataset.digest) > MAX_DATASET_DIGEST_SIZE:
        raise MlflowException(
            f"Dataset digest exceeds the maximum length of {MAX_DATASET_DIGEST_SIZE}",
            INVALID_PARAMETER_VALUE,
        )
    if len(dataset.source) > MAX_DATASET_SOURCE_SIZE:
        raise MlflowException(
            f"Dataset source exceeds the maximum length of {MAX_DATASET_SOURCE_SIZE}",
            INVALID_PARAMETER_VALUE,
        )
    if dataset.schema is not None and len(dataset.schema) > MAX_DATASET_SCHEMA_SIZE:
        raise MlflowException(
            f"Dataset schema exceeds the maximum length of {MAX_DATASET_SCHEMA_SIZE}",
            INVALID_PARAMETER_VALUE,
        )
    if dataset.profile is not None and len(dataset.profile) > MAX_DATASET_PROFILE_SIZE:
        raise MlflowException(
            f"Dataset profile exceeds the maximum length of {MAX_DATASET_PROFILE_SIZE}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_input_tags(input_tags: List[InputTag]):
    for input_tag in input_tags:
        _validate_input_tag(input_tag)


def _validate_input_tag(input_tag: InputTag):
    if input_tag is None:
        raise MlflowException("InputTag cannot be None", INVALID_PARAMETER_VALUE)
    if input_tag.key is None:
        raise MlflowException("InputTag key cannot be None", INVALID_PARAMETER_VALUE)
    if input_tag.value is None:
        raise MlflowException("InputTag value cannot be None", INVALID_PARAMETER_VALUE)
    if len(input_tag.key) > MAX_INPUT_TAG_KEY_SIZE:
        raise MlflowException(
            f"InputTag key exceeds the maximum length of {MAX_INPUT_TAG_KEY_SIZE}",
            INVALID_PARAMETER_VALUE,
        )
    if len(input_tag.value) > MAX_INPUT_TAG_VALUE_SIZE:
        raise MlflowException(
            f"InputTag value exceeds the maximum length of {MAX_INPUT_TAG_VALUE_SIZE}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_username(username):
    if username is None or username == "":
        raise MlflowException("Username cannot be empty.", INVALID_PARAMETER_VALUE)


def _validate_trace_tag(key, value):
    _validate_tag_name(key)
    key = _validate_length_limit("Trace tag key", MAX_TRACE_TAG_KEY_LENGTH, key)
    value = _validate_length_limit(
        "Trace tag value", MAX_TRACE_TAG_VAL_LENGTH, value, truncate=True
    )
    return key, value
