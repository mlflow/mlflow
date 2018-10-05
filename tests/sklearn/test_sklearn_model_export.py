from __future__ import print_function

import os
import pickle
import tempfile
import unittest

import numpy as np
import sklearn.datasets as datasets
import sklearn.linear_model as glm
import sklearn.neighbors as knn

from mlflow import sklearn, pyfunc
import mlflow
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env


def _load_pyfunc(path):
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
        old_uri = mlflow.get_tracking_uri()
        # should_start_run tests whether or not calling log_model() automatically starts a run.
        for should_start_run in [False, True]:
            with TempDir(chdr=True, remove_on_exit=True) as tmp:
                try:
                    mlflow.set_tracking_uri("test")
                    if should_start_run:
                        mlflow.start_run()
                    artifact_path = "linear"
                    conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                    _mlflow_conda_env(conda_env, additional_pip_deps=["sklearn"])
                    sklearn.log_model(sk_model=self._linear_lr,
                                      artifact_path=artifact_path,
                                      conda_env=conda_env)
                    x = sklearn.load_model(artifact_path, run_id=mlflow.active_run().info.run_uuid)
                    model_path = _get_model_log_dir(
                            artifact_path, mlflow.active_run().info.run_uuid)
                    model_config = Model.load(os.path.join(model_path, "MLmodel"))
                    assert pyfunc.FLAVOR_NAME in model_config.flavors
                    assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                    env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
                    assert os.path.exists(os.path.join(model_path, env_path))
                    xpred = x.predict(self._X)
                    np.testing.assert_array_equal(self._linear_lr_predict, xpred)
                finally:
                    mlflow.end_run()
                    mlflow.set_tracking_uri(old_uri)


if __name__ == '__main__':
    unittest.main()
