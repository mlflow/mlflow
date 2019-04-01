import time
import unittest

from mlflow.entities import Metric, RunData
from tests.helper_functions import random_str, random_int


class TestRunData(unittest.TestCase):
    def _check_metrics(self, metrics_1, metrics_2):
        for key, metric in metrics_1.items():
            self.assertIsInstance(metric, Metric)
            self.assertEqual(metric.key, key)
        metric_objs = metrics_1.values()
        expected_metric_objs = metrics_2.values()
        self.assertEqual(set(metrics_1.keys()), set(metrics_2.keys()))
        self.assertEqual(set([m.value for m in metric_objs]),
                         set([m.value for m in expected_metric_objs]))
        self.assertEqual(set([m.timestamp for m in metric_objs]),
                         set([m.timestamp for m in expected_metric_objs]))

    def _check_params(self, params_1, params_2):
        for p_key, p_val in params_1.items():
            self.assertIsInstance(p_key, str)
            self.assertIsInstance(p_val, str)
        self.assertEqual(params_1, params_2)

    def _check_tags(self, tags_1, tags_2):
        for t_key, t_val in tags_1.items():
            self.assertIsInstance(t_key, str)
            self.assertIsInstance(t_val, str)
        self.assertEqual(tags_1, tags_2)

    def _check(self, rd, metrics, params, tags):
        self.assertIsInstance(rd, RunData)
        self._check_metrics(rd.metrics, metrics)
        self._check_params(rd.params, params)
        self._check_tags(rd.tags, tags)

    @staticmethod
    def _create():
        metrics = {}
        for _ in range(100):
            key = random_str(10)
            metrics[key] = Metric(key, random_int(0, 1000),
                                  int(time.time()) + random_int(-1e4, 1e4))
        params = {random_str(10): random_str(random_int(10, 35)) for _ in range(10)}  # noqa
        tags = {random_str(10): random_str(random_int(10, 35)) for _ in range(10)}  # noqa
        rd = RunData(metrics=metrics, params=params, tags=tags)
        return rd, metrics, params, tags

    def test_creation_and_hydration(self):
        rd1, metrics, params, tags = self._create()
        self._check(rd1, metrics, params, tags)

        as_dict = {"metrics": metrics, "params": params, "tags": tags}
        self.assertEqual(dict(rd1), as_dict)

        proto = rd1.to_proto()
        rd2 = RunData.from_proto(proto)
        self._check(rd2, metrics, params, tags)

        rd3 = RunData.from_dictionary(as_dict)
        self._check(rd3, metrics, params, tags)
