import copy
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from mlflow.utils.validation import (
    _is_numeric,
    _validate_metric_name,
    _validate_param_name,
    _validate_tag_name,
    _validate_run_id,
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_experiment_artifact_location,
    _validate_db_type_string,
    _validate_experiment_name,
)

GOOD_METRIC_OR_PARAM_NAMES = [
    "a",
    "Ab-5_",
    "a/b/c",
    "a.b.c",
    ".a",
    "b.",
    "a..a/._./o_O/.e.",
    "a b/c d",
]
BAD_METRIC_OR_PARAM_NAMES = [
    "",
    ".",
    "/",
    "..",
    "//",
    "a//b",
    "a/./b",
    "/a",
    "a/",
    ":",
    "\\",
    "./",
    "/./",
]


def test_is_numeric():
    assert _is_numeric(0)
    assert _is_numeric(0.0)
    assert not _is_numeric(True)
    assert not _is_numeric(False)
    assert not _is_numeric("0")
    assert not _is_numeric(None)


def test_validate_metric_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_metric_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(MlflowException, match="Invalid metric name") as e:
            _validate_metric_name(bad_name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_param_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_param_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(MlflowException, match="Invalid parameter name") as e:
            _validate_param_name(bad_name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_tag_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_tag_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(MlflowException, match="Invalid tag name") as e:
            _validate_tag_name(bad_name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_run_id():
    for good_id in [
        "a" * 32,
        "f0" * 16,
        "abcdef0123456789" * 2,
        "a" * 33,
        "a" * 31,
        "a" * 256,
        "A" * 32,
        "g" * 32,
        "a_" * 32,
        "abcdefghijklmnopqrstuvqxyz",
    ]:
        _validate_run_id(good_id)
    for bad_id in ["a/bc" * 8, "", "a" * 400, "*" * 5]:
        with pytest.raises(MlflowException, match="Invalid run ID") as e:
            _validate_run_id(bad_id)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_batch_log_limits():
    too_many_metrics = [Metric("metric-key-%s" % i, 1, 0, i * 2) for i in range(1001)]
    too_many_params = [Param("param-key-%s" % i, "b") for i in range(101)]
    too_many_tags = [RunTag("tag-key-%s" % i, "b") for i in range(101)]

    good_kwargs = {"metrics": [], "params": [], "tags": []}
    bad_kwargs = {
        "metrics": [too_many_metrics],
        "params": [too_many_params],
        "tags": [too_many_tags],
    }
    match = r"A batch logging request can contain at most \d+"
    for arg_name, arg_values in bad_kwargs.items():
        for arg_value in arg_values:
            final_kwargs = copy.deepcopy(good_kwargs)
            final_kwargs[arg_name] = arg_value
            with pytest.raises(MlflowException, match=match):
                _validate_batch_log_limits(**final_kwargs)
    # Test the case where there are too many entities in aggregate
    with pytest.raises(MlflowException, match=match):
        _validate_batch_log_limits(too_many_metrics[:900], too_many_params[:51], too_many_tags[:50])
    # Test that we don't reject entities within the limit
    _validate_batch_log_limits(too_many_metrics[:1000], [], [])
    _validate_batch_log_limits([], too_many_params[:100], [])
    _validate_batch_log_limits([], [], too_many_tags[:100])


def test_validate_batch_log_data():
    metrics_with_bad_key = [
        Metric("good-metric-key", 1.0, 0, 0),
        Metric("super-long-bad-key" * 1000, 4.0, 0, 0),
    ]
    metrics_with_bad_val = [Metric("good-metric-key", "not-a-double-val", 0, 0)]
    metrics_with_bool_val = [Metric("good-metric-key", True, 0, 0)]
    metrics_with_bad_ts = [Metric("good-metric-key", 1.0, "not-a-timestamp", 0)]
    metrics_with_neg_ts = [Metric("good-metric-key", 1.0, -123, 0)]
    metrics_with_bad_step = [Metric("good-metric-key", 1.0, 0, "not-a-step")]
    params_with_bad_key = [
        Param("good-param-key", "hi"),
        Param("super-long-bad-key" * 1000, "but-good-val"),
    ]
    params_with_bad_val = [
        Param("good-param-key", "hi"),
        Param("another-good-key", "but-bad-val" * 1000),
    ]
    tags_with_bad_key = [
        RunTag("good-tag-key", "hi"),
        RunTag("super-long-bad-key" * 1000, "but-good-val"),
    ]
    tags_with_bad_val = [
        RunTag("good-tag-key", "hi"),
        RunTag("another-good-key", "but-bad-val" * 1000),
    ]
    bad_kwargs = {
        "metrics": [
            metrics_with_bad_key,
            metrics_with_bad_val,
            metrics_with_bool_val,
            metrics_with_bad_ts,
            metrics_with_neg_ts,
            metrics_with_bad_step,
        ],
        "params": [params_with_bad_key, params_with_bad_val],
        "tags": [tags_with_bad_key, tags_with_bad_val],
    }
    good_kwargs = {"metrics": [], "params": [], "tags": []}
    for arg_name, arg_values in bad_kwargs.items():
        for arg_value in arg_values:
            final_kwargs = copy.deepcopy(good_kwargs)
            final_kwargs[arg_name] = arg_value
            with pytest.raises(MlflowException, match=r".+"):
                _validate_batch_log_data(**final_kwargs)
    # Test that we don't reject entities within the limit
    _validate_batch_log_data(
        metrics=[Metric("metric-key", 1.0, 0, 0)],
        params=[Param("param-key", "param-val")],
        tags=[RunTag("tag-key", "tag-val")],
    )


def test_validate_experiment_artifact_location():
    _validate_experiment_artifact_location("abcde")
    _validate_experiment_artifact_location(None)
    with pytest.raises(MlflowException, match="Artifact location cannot be a runs:/ URI"):
        _validate_experiment_artifact_location("runs:/blah/bleh/blergh")


def test_validate_experiment_name():
    _validate_experiment_name("validstring")
    bytestring = b"test byte string"
    _validate_experiment_name(bytestring.decode("utf-8"))
    for invalid_name in ["", 12, 12.7, None, {}, []]:
        with pytest.raises(MlflowException, match="Invalid experiment name"):
            _validate_experiment_name(invalid_name)


def test_validate_list_experiments_max_results():
    client = mlflow.tracking.MlflowClient()
    client.list_experiments(max_results=50)
    with pytest.raises(MlflowException, match="It must be at most 50000"):
        client.list_experiments(max_results=50001)
    for invalid_num in [-12, 0]:
        with pytest.raises(MlflowException, match="It must be at least 1"):
            client.list_experiments(max_results=invalid_num)


def test_db_type():
    for db_type in ["mysql", "mssql", "postgresql", "sqlite"]:
        # should not raise an exception
        _validate_db_type_string(db_type)

    # error cases
    for db_type in ["MySQL", "mongo", "cassandra", "sql", ""]:
        with pytest.raises(MlflowException, match="Invalid database engine") as e:
            _validate_db_type_string(db_type)
        assert "Invalid database engine" in e.value.message
