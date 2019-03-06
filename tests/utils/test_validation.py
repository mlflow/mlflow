import copy
import pytest

from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.validation import _validate_metric_name, _validate_param_name, \
                                    _validate_tag_name, _validate_run_id, \
                                    _validate_batch_log_data, _validate_batch_log_limits

GOOD_METRIC_OR_PARAM_NAMES = [
    "a", "Ab-5_", "a/b/c", "a.b.c", ".a", "b.", "a..a/._./o_O/.e.", "a b/c d",
]
BAD_METRIC_OR_PARAM_NAMES = [
    "", ".", "/", "..", "//", "a//b", "a/./b", "/a", "a/", ":", "\\", "./", "/./",
]


def test_validate_metric_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_metric_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(Exception, match="Invalid metric name"):
            _validate_metric_name(bad_name)


def test_validate_param_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_param_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(Exception, match="Invalid parameter name"):
            _validate_param_name(bad_name)


def test_validate_tag_name():
    for good_name in GOOD_METRIC_OR_PARAM_NAMES:
        _validate_tag_name(good_name)
    for bad_name in BAD_METRIC_OR_PARAM_NAMES:
        with pytest.raises(Exception, match="Invalid tag name"):
            _validate_tag_name(bad_name)


def test_validate_run_id():
    for good_id in ["a" * 32, "f0" * 16, "abcdef0123456789" * 2]:
        _validate_run_id(good_id)
    for bad_id in ["a" * 33, "a" * 31, "A" * 32, "g" * 32, "a/bc" * 8, "_" * 32]:
        with pytest.raises(Exception, match="Invalid run ID"):
            _validate_run_id(bad_id)


def test_validate_batch_log_limits():
    too_many_metrics = [Metric("metric-key-%s" % i, 1, 0) for i in range(1001)]
    too_many_params = [Param("param-key-%s" % i, "b") for i in range(101)]
    too_many_tags = [RunTag("tag-key-%s" % i, "b") for i in range(101)]

    good_kwargs = {"metrics": [], "params": [], "tags": []}
    bad_kwargs = {
        "metrics": [too_many_metrics],
        "params": [too_many_params],
        "tags": [too_many_tags],
    }
    for arg_name, arg_values in bad_kwargs.items():
        for arg_value in arg_values:
            final_kwargs = copy.deepcopy(good_kwargs)
            final_kwargs[arg_name] = arg_value
            print(arg_value, len(arg_value))
            with pytest.raises(MlflowException):
                _validate_batch_log_limits(**final_kwargs)
    # Test the case where there are too many entities in aggregate
    with pytest.raises(MlflowException):
        _validate_batch_log_limits(too_many_metrics[:900], too_many_params[:51],
                                   too_many_tags[:50])
    # Test that we don't reject entities within the limit
    _validate_batch_log_limits(too_many_metrics[:1000], [], [])
    _validate_batch_log_limits([], too_many_params[:100], [])
    _validate_batch_log_limits([], [], too_many_tags[:100])


def test_validate_batch_log_data():
    metrics_with_bad_key = [Metric("good-metric-key", 1.0, 0),
                            Metric("super-long-bad-key" * 1000, 4.0, 0)]
    params_with_bad_key = [Param("good-param-key", "hi"),
                           Param("super-long-bad-key" * 1000, "but-good-val")]
    params_with_bad_val = [Param("good-param-key", "hi"),
                           Param("another-good-key", "but-bad-val" * 1000)]
    tags_with_bad_key = [RunTag("good-tag-key", "hi"),
                         RunTag("super-long-bad-key" * 1000, "but-good-val")]
    tags_with_bad_val = [RunTag("good-tag-key", "hi"),
                         RunTag("another-good-key", "but-bad-val" * 1000)]
    overwriting_param = [Param("key", "val"), Param("key", "different-val")]
    bad_kwargs = {
        "metrics": [metrics_with_bad_key],
        "params": [params_with_bad_key, params_with_bad_val, overwriting_param],
        "tags": [tags_with_bad_key, tags_with_bad_val],
    }
    good_kwargs = {"metrics": [], "params": [], "tags": []}
    for arg_name, arg_values in bad_kwargs.items():
        for arg_value in arg_values:
            final_kwargs = copy.deepcopy(good_kwargs)
            final_kwargs[arg_name] = arg_value
            with pytest.raises(MlflowException):
                _validate_batch_log_data(**final_kwargs)
