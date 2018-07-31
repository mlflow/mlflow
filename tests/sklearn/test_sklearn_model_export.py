from __future__ import print_function

import pickle
import tempfile
import unittest

import numpy as np
import sklearn.datasets as datasets
import sklearn.linear_model as glm
import sklearn.neighbors as knn

from mlflow import sklearn, pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._knn = knn.KNeighborsClassifier()
        self._knn.fit(self._X, self._y)
        self._knn_predict = self._knn.predict(self._X)
        self._linear_lr = glm.LogisticRegression()
        self._linear_lr.fit(self._X, self._y)
        self._linear_lr_predict = self._linear_lr.predict(self._X)

    def test_model_save_load(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")
            sklearn.save_model(self._knn, path=path)
            x = sklearn.load_model(path)
            xpred = x.predict(self._X)
            np.testing.assert_array_equal(self._knn_predict, xpred)
            # sklearn should also be stored as a valid pyfunc model
            # test pyfunc compatibility
            y = pyfunc.load_pyfunc(path)
            ypred = y.predict(self._X)
            np.testing.assert_array_equal(self._knn_predict, ypred)

    def test_model_log(self):
        old_uri = tracking.get_tracking_uri()
        # should_start_run tests whether or not calling log_model() automatically starts a run.
        for should_start_run in [False, True]:
            with TempDir(chdr=True, remove_on_exit=True) as tmp:
                try:
                    tracking.set_tracking_uri("test")
                    if should_start_run:
                        tracking.start_run()
                    sklearn.log_model(sk_model=self._linear_lr, artifact_path="linear")
                    x = sklearn.load_model("linear", run_id=tracking.active_run().info.run_uuid)
                    xpred = x.predict(self._X)
                    np.testing.assert_array_equal(self._linear_lr_predict, xpred)
                finally:
                    tracking.end_run()
                    tracking.set_tracking_uri(old_uri)


if __name__ == '__main__':
    unittest.main()
