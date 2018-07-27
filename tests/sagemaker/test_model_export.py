from __future__ import print_function

import os
import pickle
import tempfile
import unittest

import sklearn.datasets as datasets
import sklearn.linear_model as glm

import numpy as np
import pandas as pd
import pytest

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc

from mlflow.utils.environment import _mlflow_conda_env

from tests.helper_functions import score_model_in_sagemaker_docker_container


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

    @pytest.mark.large
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
                pyfunc.save_model(input_path, loader_module="mlflow.sklearn",
                                  data_path=model_pkl,
                                  conda_env=_mlflow_conda_env(tmp.path(conda_env)))
                xpred = score_model_in_sagemaker_docker_container(input_path, self._iris_df)
                print('expected', self._linear_lr_predict)
                print('actual  ', xpred)
                np.testing.assert_array_equal(self._linear_lr_predict, xpred)
        finally:
            if path_to_remove:
                try:
                    import shutil
                    shutil.rmtree(path_to_remove)
                except OSError:
                    print("Failed to remove", path_to_remove)


if __name__ == '__main__':
    unittest.main()
