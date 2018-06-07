from __future__ import print_function

import os
import pickle
from subprocess import Popen, PIPE, STDOUT
import tempfile
import unittest

import sklearn.datasets as datasets
import sklearn.linear_model as glm

import numpy as np
import pandas as pd

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc

from mlflow.sagemaker import cli, DEV_FLAG


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)




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
        os.environ[DEV_FLAG] = "1"
        os.environ["LC_ALL"]= "en_US.UTF-8"
        os.environ["LANG"] = "en_US.UTF-8"
        self._build_container()

    def _build_container(self):
        from mlflow.sagemaker import build_container
        build_container()

    def test_model_export(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            model_pkl = tmp.path("model.pkl")
            with open(model_pkl, "wb") as f:
                pickle.dump(self._linear_lr, f)
            input_path = tmp.path("input_model")
            pyfunc.save_model(input_path, loader_module="test_model_export", code_path=[__file__],
                              data_path=model_pkl)

            proc = Popen(['mlflow', 'sagemaker', 'run-local', '-m', input_path], stdout=PIPE,
                         stderr=STDOUT, universal_newlines=True)

            for x in iter(proc.stdout.readline, ""):
                print(x)
                if "Booting worker with pid" in x:
                    break
            self.assertTrue(proc.poll() is None, "scoring process died")
            import requests
            print("curl data in")
            x = self._iris_df.to_dict(orient='records')
            y = requests.post(url='http://localhost:5000/invocations', json=x)
            xpred = [int(z) for z in y.text.split("\n")[1:-1]]
            np.testing.assert_array_equal(self._linear_lr_predict, xpred)

if __name__ == '__main__':
    unittest.main()
