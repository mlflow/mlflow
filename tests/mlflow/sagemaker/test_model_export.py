from __future__ import print_function

import os
import pickle
import requests
from subprocess import Popen, PIPE, STDOUT
import tempfile
import time
import unittest

import sklearn.datasets as datasets
import sklearn.linear_model as glm

import numpy as np
import pandas as pd

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc

import mlflow.sagemaker


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


CONDA_ENV = """
name: mlflow-env
channels:
  - anaconda
  - defaults
dependencies:
  - python={python_version}

"""


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._iris_df = pd.DataFrame(self._X, columns=iris.feature_names[:2])
        self._linear_lr = glm.LogisticRegression()
        self._linear_lr.fit(self._X, self._y)
        self._linear_lr_predict = self._linear_lr.predict(self._X)
        os.environ["LC_ALL"] = "en_US.UTF-8"
        os.environ["LANG"] = "en_US.UTF-8"
        mlflow_root = os.environ.get("MLFLOW_HOME") if "MLFLOW_HOME" in os.environ \
            else os.path.dirname(os.path.dirname(os.path.abspath(mlflow.__file__)))
        # "/home/travis/build/databricks/mlflow"
        print("Building mlflow Docker image with MLFLOW_HOME =", mlflow_root)
        mlflow.sagemaker.build_image(mlflow_home=mlflow_root)

    def test_model_export(self):
        path_to_remove = None
        try:
            with TempDir(chdr=True, remove_on_exit=False) as tmp:
                path_to_remove = tmp._path
                # NOTE: Changed dir to temp dir and use relative paths to get around the way temp
                # dirs are handled in python.
                model_pkl = tmp.path("model.pkl")
                with open(model_pkl, "wb") as f:
                    pickle.dump(self._linear_lr, f)
                input_path = tmp.path("input_model")
                conda_env = "conda.env"
                from sys import version_info
                python_version = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                                  minor=version_info.minor,
                                                                  micro=version_info.micro)
                with open(conda_env, "w") as f:
                    f.write(CONDA_ENV.format(python_version=python_version))
                pyfunc.save_model(input_path, loader_module="test_model_export",
                                  code_path=[__file__],
                                  data_path=model_pkl,
                                  conda_env=conda_env)
                proc = Popen(['mlflow', 'sagemaker', 'run-local', '-m', input_path], stdout=PIPE,
                             stderr=STDOUT, universal_newlines=True)

                try:
                    for i in range(0, 50):
                        self.assertTrue(proc.poll() is None, "scoring process died")
                        time.sleep(5)
                        # noinspection PyBroadException
                        try:
                            ping_status = requests.get(url='http://localhost:5000/ping')
                            print('connection attempt', i, "server is up! ping status", ping_status)
                            if ping_status.status_code == 200:
                                break
                        except Exception:
                            print('connection attempt', i, "failed, server is not up yet")

                    self.assertTrue(proc.poll() is None, "scoring process died")
                    ping_status = requests.get(url='http://localhost:5000/ping')
                    print("server up, ping status", ping_status)
                    if ping_status.status_code != 200:
                        raise Exception("ping failed, server is not happy")
                    x = self._iris_df.to_dict(orient='records')
                    y = requests.post(url='http://localhost:5000/invocations', json=x)
                    import json
                    xpred = json.loads(y.content)
                    print('expected', self._linear_lr_predict)
                    print('actual  ', xpred)
                    np.testing.assert_array_equal(self._linear_lr_predict, xpred)

                finally:
                    if proc.poll() is None:
                        proc.terminate()
                    print("captured output of the scoring process")
                    print(proc.stdout.read())
        finally:
            if path_to_remove:
                try:
                    import shutil
                    shutil.rmtree(path_to_remove)
                except PermissionError:
                    print("Failed to remove", path_to_remove)


if __name__ == '__main__':
    unittest.main()
