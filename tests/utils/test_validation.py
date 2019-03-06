import pytest

from mlflow.utils.validation import _validate_metric_name, _validate_param_name, \
                                    _validate_tag_name, _validate_run_id

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
