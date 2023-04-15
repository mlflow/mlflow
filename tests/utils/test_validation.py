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
    _validate_model_alias_name,
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

GOOD_ALIAS_NAMES = [
    "a",
    "Ab-5_",
    "test-alias",
    "1a2b5cDeFgH",
    "a" * 256,
    "lates",
    "v123_temp",
    "123",
    "123v",
    "temp_V123",
]

BAD_ALIAS_NAMES = [
    "",
    ".",
    "/",
    "..",
    "//",
    "a b",
    "a/./b",
    "/a",
    "a/",
    ":",
    "\\",
    "./",
    "/./",
    "a" * 257,
    None,
    "$dgs",
    "latest",
    "Latest",
    "v123",
    "V1",
]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a", False),
        ("a/b/c", False),
        ("a.b/c", False),
        (".a", False),
        # Not unique paths
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
        # Non-numeric cases
        (True, False),
        (False, False),
        ("0", False),
        (None, False),
    ],
)
def test_is_numeric(value, expected):
    assert _is_numeric(value) is expected


@pytest.mark.parametrize("metric_name", GOOD_METRIC_OR_PARAM_NAMES)
def test_validate_metric_name_good(metric_name):
    _validate_metric_name(metric_name)


@pytest.mark.parametrize("metric_name", BAD_METRIC_OR_PARAM_NAMES)
def test_validate_metric_name_bad(metric_name):
    with pytest.raises(MlflowException, match="Invalid metric name") as e:
        _validate_metric_name(metric_name)
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("param_name", GOOD_METRIC_OR_PARAM_NAMES)
def test_validate_param_name_good(param_name):
    _validate_param_name(param_name)


@pytest.mark.parametrize("param_name", BAD_METRIC_OR_PARAM_NAMES)
def test_validate_param_name_bad(param_name):
    with pytest.raises(MlflowException, match="Invalid parameter name") as e:
        _validate_param_name(param_name)
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("tag_name", GOOD_METRIC_OR_PARAM_NAMES)
def test_validate_tag_name_good(tag_name):
    _validate_tag_name(tag_name)


@pytest.mark.parametrize("tag_name", BAD_METRIC_OR_PARAM_NAMES)
def test_validate_tag_name_bad(tag_name):
    with pytest.raises(MlflowException, match="Invalid tag name") as e:
        _validate_tag_name(tag_name)
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("alias_name", GOOD_ALIAS_NAMES)
def test_validate_model_alias_name_good(alias_name):
    _validate_model_alias_name(alias_name)


@pytest.mark.parametrize("alias_name", BAD_ALIAS_NAMES)
def test_validate_model_alias_name_bad(alias_name):
    with pytest.raises(MlflowException, match="alias name") as e:
        _validate_model_alias_name(alias_name)
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    "run_id",
    [
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
    ],
)
def test_validate_run_id_good(run_id):
    _validate_run_id(run_id)


@pytest.mark.parametrize("run_id", ["a/bc" * 8, "", "a" * 400, "*" * 5])
def test_validate_run_id_bad(run_id):
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


@pytest.mark.parametrize("location", ["abcde", None])
def test_validate_experiment_artifact_location_good(location):
    _validate_experiment_artifact_location(location)


@pytest.mark.parametrize("location", ["runs:/blah/bleh/blergh"])
def test_validate_experiment_artifact_location_bad(location):
    with pytest.raises(MlflowException, match="Artifact location cannot be a runs:/ URI"):
        _validate_experiment_artifact_location(location)


@pytest.mark.parametrize("experiment_name", ["validstring", b"test byte string".decode("utf-8")])
def test_validate_experiment_name_good(experiment_name):
    _validate_experiment_name(experiment_name)


@pytest.mark.parametrize("experiment_name", ["", 12, 12.7, None, {}, []])
def test_validate_experiment_name_bad(experiment_name):
    with pytest.raises(MlflowException, match="Invalid experiment name"):
        _validate_experiment_name(experiment_name)


@pytest.mark.parametrize("db_type", ["mysql", "mssql", "postgresql", "sqlite"])
def test_validate_db_type_string_good(db_type):
    _validate_db_type_string(db_type)


@pytest.mark.parametrize("db_type", ["MySQL", "mongo", "cassandra", "sql", ""])
def test_validate_db_type_string_bad(db_type):
    with pytest.raises(MlflowException, match="Invalid database engine") as e:
        _validate_db_type_string(db_type)
    assert "Invalid database engine" in e.value.message
