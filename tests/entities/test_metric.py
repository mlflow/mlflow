import time

from mlflow.entities import Metric
from tests.helper_functions import random_str, random_int


def _check(metric, key, value, timestamp, step):
    assert type(metric) == Metric
    assert metric.key == key
    assert metric.value == value
    assert metric.timestamp == timestamp
    assert metric.step == step


def test_creation_and_hydration():
    key = random_str()
    value = 10000
    ts = int(1000 * time.time())
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
