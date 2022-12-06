import copy
import pytest
from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from mlflow.utils.validation import (
    path_not_unique,
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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a", False),
        ("a/b/c", False),
        ("a.b/c", False),
        (".a", False),
        ("./a", True),
        ("a/b/../c", True),
        (".", True),
        ("../a/b", True),
        ("/a/b/c", True),
    ],
)
def test_path_not_unique(path, expected):
    assert path_not_unique(path) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (0.0, True),
        # error cases
        (True, False),
        (False, False),
        ("0", False),
        (None, False),
    ],
)
def test_is_numeric(value, expected):
    assert _is_numeric(value) is expected


@pytest.mark.parametrize(
    ("name", "is_good_name"),
    [
        *[(name, True) for name in GOOD_METRIC_OR_PARAM_NAMES],
        # error cases
        *[(name, False) for name in BAD_METRIC_OR_PARAM_NAMES],
    ],
)
def test_validate_metric_name(name, is_good_name):
    if is_good_name:
        _validate_metric_name(name)
    else:
        with pytest.raises(MlflowException, match="Invalid metric name") as e:
            _validate_metric_name(name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    ("name", "is_good_name"),
    [
        *[(name, True) for name in GOOD_METRIC_OR_PARAM_NAMES],
        # error cases
        *[(name, False) for name in BAD_METRIC_OR_PARAM_NAMES],
    ],
)
def test_validate_param_name(name, is_good_name):
    if is_good_name:
        _validate_param_name(name)
    else:
        with pytest.raises(MlflowException, match="Invalid parameter name") as e:
            _validate_param_name(name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    ("name", "is_good_name"),
    [
        *[(name, True) for name in GOOD_METRIC_OR_PARAM_NAMES],
        # error cases
        *[(name, False) for name in BAD_METRIC_OR_PARAM_NAMES],
    ],
)
def test_validate_tag_name(name, is_good_name):
    if is_good_name:
        _validate_tag_name(name)
    else:
        with pytest.raises(MlflowException, match="Invalid tag name") as e:
            _validate_tag_name(name)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    ("run_id", "is_good_id"),
    [
        ("a" * 32, True),
        ("f0" * 16, True),
        ("abcdef0123456789" * 2, True),
        ("a" * 33, True),
        ("a" * 31, True),
        ("a" * 256, True),
        ("A" * 32, True),
        ("g" * 32, True),
        ("a_" * 32, True),
        ("abcdefghijklmnopqrstuvqxyz", True),
        # error cases
        ("a/bc" * 8, False),
        ("", False),
        ("a" * 400, False),
        ("*" * 5, False),
    ],
)
def test_validate_run_id(run_id, is_good_id):
    if is_good_id:
        _validate_run_id(run_id)
    else:
        with pytest.raises(MlflowException, match="Invalid run ID") as e:
            _validate_run_id(run_id)
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


@pytest.mark.parametrize(
    ("location", "is_good_location"),
    [
        ("abcde", True),
        (None, True),
        # error cases
        ("runs:/blah/bleh/blergh", False),
    ],
)
def test_validate_experiment_artifact_location(location, is_good_location):
    if is_good_location:
        _validate_experiment_artifact_location(location)
    else:
        with pytest.raises(MlflowException, match="Artifact location cannot be a runs:/ URI"):
            _validate_experiment_artifact_location(location)


@pytest.mark.parametrize(
    ("experiment_name", "is_good_name"),
    [
        ("validstring", True),
        (b"test byte string".decode("utf-8"), True),
        # error cases
        ("", False),
        (12, False),
        (12.7, False),
        (None, False),
        ({}, False),
        ([], False),
    ],
)
def test_validate_experiment_name(experiment_name, is_good_name):
    if is_good_name:
        _validate_experiment_name(experiment_name)
    else:
        with pytest.raises(MlflowException, match="Invalid experiment name"):
            _validate_experiment_name(experiment_name)


@pytest.mark.parametrize(
    ("db_type", "is_good_type"),
    [
        ("mysql", True),
        ("mssql", True),
        ("postgresql", True),
        ("sqlite", True),
        # error cases
        ("MySQL", False),
        ("mongo", False),
        ("cassandra", False),
        ("sql", False),
        ("", False),
    ],
)
def test_db_type(db_type, is_good_type):
    if is_good_type:
        _validate_db_type_string(db_type)
    else:
        with pytest.raises(MlflowException, match="Invalid database engine") as e:
            _validate_db_type_string(db_type)
        assert "Invalid database engine" in e.value.message
