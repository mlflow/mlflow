import time
import unittest

from mlflow.entities import Metric, Param, RunData, RunTag
from tests.helper_functions import random_str, random_int


class TestRunData(unittest.TestCase):
    def _check_metrics(self, metrics_1, metrics_2):
        for metric in metrics_1:
            self.assertIsInstance(metric, Metric)
        self.assertEqual(set([m.key for m in metrics_1]), set([m.key for m in metrics_2]))
        self.assertEqual(set([m.value for m in metrics_1]), set([m.value for m in metrics_2]))
        self.assertEqual(set([m.timestamp for m in metrics_1]),
                         set([m.timestamp for m in metrics_2]))

    def _check_params(self, params_1, params_2):
        for param in params_1:
            self.assertIsInstance(param, Param)
        self.assertEqual(set([p.key for p in params_1]), set([p.key for p in params_2]))
        self.assertEqual(set([p.value for p in params_1]), set([p.value for p in params_2]))

    def _check_tags(self, tags_1, tags_2):
        for tag in tags_1:
            self.assertIsInstance(tag, RunTag)
        self.assertEqual(set([t.key for t in tags_1]), set([t.key for t in tags_2]))
        self.assertEqual(set([t.value for t in tags_2]), set([t.value for t in tags_2]))

    def _check(self, rd, metrics, params, tags):
        self.assertIsInstance(rd, RunData)
        self._check_metrics(rd.metrics, metrics)
        self._check_params(rd.params, params)
        self._check_tags(rd.tags, tags)

    @staticmethod
    def _create():
        metrics = [Metric(random_str(10), random_int(0, 1000),
                          int(time.time() + random_int(-1e4, 1e4)))
                   for _ in range(100)]
        params = [Param(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]  # noqa
        tags = [RunTag(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]  # noqa
        rd = RunData()
        for p in params:
            rd._add_param(p)
        for m in metrics:
            rd._add_metric(m)
        for t in tags:
            rd._add_tag(t)
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
