import time
import unittest

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_data import RunData
from tests.helper_functions import random_str, random_int


class TestRunData(unittest.TestCase):
    def _check_metrics(self, metrics_1, metrics_2):
        self.assertEqual(set([m.key for m in metrics_1]), set([m.key for m in metrics_2]))
        self.assertEqual(set([m.value for m in metrics_1]), set([m.value for m in metrics_2]))
        self.assertEqual(set([m.timestamp for m in metrics_1]),
                         set([m.timestamp for m in metrics_2]))

    def _check_params(self, params_1, params_2):
        self.assertEqual(set([p.key for p in params_1]), set([p.key for p in params_2]))
        self.assertEqual(set([p.value for p in params_1]), set([p.value for p in params_2]))

    def _check(self, rd, metrics, params):
        self._check_metrics(rd.metrics, metrics)
        self._check_params(rd.params, params)

    @staticmethod
    def _create():
        metrics = [Metric(random_str(10), random_int(), int(time.time() + random_int(-1e4, 1e4)))
                   for _ in range(100)]
        params = [Param(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]  # noqa
        rd = RunData()
        for p in params:
            rd.add_param(p)
        for m in metrics:
            rd.add_metric(m)
        return rd, metrics, params

    def test_creation_and_hydration(self):
        rd1, metrics, params = self._create()
        self._check(rd1, metrics, params)

        as_dict = {"metrics": metrics, "params": params}
        self.assertEqual(dict(rd1), as_dict)

        proto = rd1.to_proto()
        rd2 = RunData.from_proto(proto)
        self._check(rd2, metrics, params)

        rd3 = RunData.from_dictionary(as_dict)
        self._check(rd3, metrics, params)
