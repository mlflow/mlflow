# pep8: disable=E501

from __future__ import print_function

import collections
import os
import pandas
import shutil
import unittest

import sklearn.datasets as datasets
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import tempfile
import mlflow.h2o
import mlflow
from mlflow import pyfunc
from mlflow.utils.file_utils import TempDir


class TestModelExport(unittest.TestCase):
    def setUp(self):
        h2o.init()
        iris = datasets.load_iris()
        data = h2o.H2OFrame({
            'feature1': list(iris.data[:, 0]),
            'feature2': list(iris.data[:, 1]),
            'target': list(map(lambda i: "Flower %d" % i, iris.target))
        })
        train, self.test = data.split_frame(ratios=[.7])

        self.gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=6)
        self.gbm.train(['feature1', 'feature2'], 'target', training_frame=train)
        self.predicted = self.gbm.predict(self.test).as_data_frame()

    def test_model_save_load(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            path = tmp.path("model")
            mlflow.h2o.save_model(self.gbm, path)

            # Loading h2o model
            gbm_loaded = mlflow.h2o.load_model(path)
            assert all(gbm_loaded.predict(self.test).as_data_frame() == self.predicted)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(path)
            assert all(pyfunc_loaded.predict(self.test.as_data_frame()) == self.predicted)

    def test_model_log(self):
        old_uri = mlflow.get_tracking_uri()
        # should_start_run tests whether or not calling log_model() automatically starts a run.
        for should_start_run in [False, True]:
            with TempDir(chdr=True, remove_on_exit=True) as tmp:
                try:
                    mlflow.set_tracking_uri("test")
                    if should_start_run:
                        mlflow.start_run()
                    mlflow.h2o.log_model(self.gbm, artifact_path="gbm")

                    # Load model
                    gbm_loaded = mlflow.h2o.load_model("gbm",
                                                       run_id=mlflow.active_run().info.run_uuid)
                    assert all(gbm_loaded.predict(self.test).as_data_frame() == self.predicted)
                finally:
                    mlflow.end_run()
                    mlflow.set_tracking_uri(old_uri)
