import time
import unittest

from mlflow.entities import Metric
from tests.helper_functions import random_str


class TestMetric(unittest.TestCase):
    def _check(self, metric, key, value, timestamp):
        self.assertIsInstance(metric, Metric)
        self.assertEqual(metric.key, key)
        self.assertEqual(metric.value, value)
        self.assertEqual(metric.timestamp, timestamp)

    def test_creation_and_hydration(self):
        key = random_str()
        value = 10000
        ts = int(time.time())

        metric = Metric(key, value, ts)
        self._check(metric, key, value, ts)

        as_dict = {"key": key, "value": value, "timestamp": ts}
        self.assertEqual(dict(metric), as_dict)

        proto = metric.to_proto()
        metric2 = metric.from_proto(proto)
        self._check(metric2, key, value, ts)

        metric3 = Metric.from_dictionary(as_dict)
        self._check(metric3, key, value, ts)
