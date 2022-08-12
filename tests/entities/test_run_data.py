import time
import unittest

from mlflow.entities import Metric, RunData, Param, RunTag
from tests.helper_functions import random_str, random_int


class TestRunData(unittest.TestCase):
    def _check_metrics(self, metric_objs, metrics_dict, expected_metrics):
        self.assertEqual(set([m.key for m in metric_objs]), set([m.key for m in expected_metrics]))
        self.assertEqual(
            set([m.value for m in metric_objs]), set([m.value for m in expected_metrics])
        )
        self.assertEqual(
            set([m.timestamp for m in metric_objs]), set([m.timestamp for m in expected_metrics])
        )
        self.assertEqual(
            set([m.step for m in metric_objs]), set([m.step for m in expected_metrics])
        )
        assert len(metrics_dict) == len(expected_metrics)
        assert metrics_dict == {m.key: m.value for m in expected_metrics}

    def _check_params(self, params_dict, expected_params):
        self.assertEqual(params_dict, {p.key: p.value for p in expected_params})

    def _check_tags(self, tags_dict, expected_tags):
        self.assertEqual(tags_dict, {t.key: t.value for t in expected_tags})

    def _check(self, rd, metrics, params, tags):
        self.assertIsInstance(rd, RunData)
        self._check_metrics(rd._metric_objs, rd.metrics, metrics)
        self._check_params(rd.params, params)
        self._check_tags(rd.tags, tags)

    @staticmethod
    def _create():
        metrics = [
            Metric(
                key=random_str(10),
                value=random_int(0, 1000),
                timestamp=int(time.time()) + random_int(-1e4, 1e4),
                step=random_int(),
            )
        ]
        params = [Param(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]
        tags = [RunTag(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]
        rd = RunData(metrics=metrics, params=params, tags=tags)
        return rd, metrics, params, tags

    def test_creation_and_hydration(self):
        rd1, metrics, params, tags = self._create()
        self._check(rd1, metrics, params, tags)

        as_dict = {
            "metrics": {m.key: m.value for m in metrics},
            "params": {p.key: p.value for p in params},
            "tags": {t.key: t.value for t in tags},
        }
        self.assertEqual(dict(rd1), as_dict)
        proto = rd1.to_proto()
        rd2 = RunData.from_proto(proto)
        self._check(rd2, metrics, params, tags)
