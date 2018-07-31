from __future__ import print_function

import unittest

import sklearn.datasets as datasets
import torch

import tempfile
import mlflow.h2o
from mlflow import pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


class TestModelExport(unittest.TestCase):
    def setUp(self):

        # Setup dataset

        # h2o.init()
        # iris = datasets.load_iris()
        # data = h2o.H2OFrame({
        #     'feature1': list(iris.data[:, 0]),
        #     'feature2': list(iris.data[:, 1]),
        #     'target': list(map(lambda i: "Flower %d" % i, iris.target))
        # })
        # train, self.test = data.split_frame(ratios=[.7])
        self.test = None

        # Setup model
        self.model = None

        # Train
        # self.model.train(['feature1', 'feature2'], 'target', training_frame=train)
        # Predict
        # self.predicted = self.gbm.predict(self.test).as_data_frame()
        self.predicted = None

    def test_log_model(self):
        pass
        # old_uri = tracking.get_tracking_uri()
        # # should_start_run tests whether or not calling log_model() automatically starts a run.
        # for should_start_run in [False, True]:
        #     with TempDir(chdr=True, remove_on_exit=True) as tmp:
        #         try:
        #             tracking.set_tracking_uri("test")
        #             if should_start_run:
        #                 tracking.start_run()
        #             mlflow.h2o.log_model(self.gbm, artifact_path="gbm")
        #
        #             # Load model
        #             gbm_loaded = mlflow.h2o.load_model("gbm",
        #                                                run_id=tracking.active_run().info.run_uuid)
        #             assert all(gbm_loaded.predict(self.test).as_data_frame() == self.predicted)
        #         finally:
        #             tracking.end_run()
        #             tracking.set_tracking_uri(old_uri)

    def test_save_and_load_model(self):
        pass
        # with TempDir(chdr=True, remove_on_exit=True) as tmp:
        #     path = tmp.path("model")
        #     mlflow.h2o.save_model(self.gbm, path)
        #
        #     # Loading h2o model
        #     gbm_loaded = mlflow.h2o.load_model(path)
        #     assert all(gbm_loaded.predict(self.test).as_data_frame() == self.predicted)
        #
        #     # Loading pyfunc model
        #     pyfunc_loaded = mlflow.pyfunc.load_pyfunc(path)
        #     assert all(pyfunc_loaded.predict(self.test.as_data_frame()) == self.predicted)

