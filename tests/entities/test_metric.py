import re

import pytest

from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from mlflow.utils.time import get_current_time_millis

from tests.helper_functions import random_int, random_str


def _check(metric, key, value, timestamp, step):
    assert type(metric) == Metric
    assert metric.key == key
    assert metric.value == value
    assert metric.timestamp == timestamp
    assert metric.step == step


def test_creation_and_hydration():
    key = random_str()
    value = 10000
    ts = get_current_time_millis()
    step = random_int()

    metric = Metric(key, value, ts, step)
    _check(metric, key, value, ts, step)

    as_dict = {"key": key, "value": value, "timestamp": ts, "step": step}
    assert dict(metric) == as_dict

    proto = metric.to_proto()
    metric2 = metric.from_proto(proto)
    _check(metric2, key, value, ts, step)

    metric3 = Metric.from_dictionary(as_dict)
    _check(metric3, key, value, ts, step)


def test_metric_to_from_dictionary():
    # Create a Metric object
    original_metric = Metric(key="accuracy", value=0.95, timestamp=1623079352000, step=1)

    # Convert the Metric object to a dictionary
    metric_dict = original_metric.to_dictionary()

    # Verify the dictionary representation
    expected_dict = {
        "key": "accuracy",
        "value": 0.95,
        "timestamp": 1623079352000,
        "step": 1,
    }
    assert metric_dict == expected_dict

    # Create a new Metric object from the dictionary
    recreated_metric = Metric.from_dictionary(metric_dict)

    # Verify the recreated Metric object matches the original
    assert recreated_metric == original_metric
    assert recreated_metric.key == original_metric.key
    assert recreated_metric.value == original_metric.value
    assert recreated_metric.timestamp == original_metric.timestamp
    assert recreated_metric.step == original_metric.step


def test_metric_from_dictionary_missing_keys():
    # Dictionary with missing keys
    incomplete_dict = {
        "key": "accuracy",
        "value": 0.95,
        "timestamp": 1623079352000,
    }

    with pytest.raises(
        MlflowException, match=re.escape("Missing required keys ['step'] in metric dictionary")
    ):
        Metric.from_dictionary(incomplete_dict)

    # Another dictionary with different missing keys
    another_incomplete_dict = {
        "key": "accuracy",
        "step": 1,
    }

    with pytest.raises(
        MlflowException,
        match=re.escape("Missing required keys ['value', 'timestamp'] in metric dictionary"),
    ):
        Metric.from_dictionary(another_incomplete_dict)
