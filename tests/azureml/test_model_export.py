from __future__ import print_function

import importlib
import os
import pickle
import tempfile
import unittest

import sklearn.datasets as datasets
import sklearn.linear_model as glm
from click.testing import CliRunner

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.azureml import cli


def _load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._linear_lr = glm.LogisticRegression()
        self._linear_lr.fit(self._X, self._y)
        self._linear_lr_predict = self._linear_lr.predict(self._X)

    def test_model_export(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            model_pkl = tmp.path("model.pkl")
            with open(model_pkl, "wb") as f:
                pickle.dump(self._linear_lr, f)
            input_path = tmp.path("input_model")
            pyfunc.save_model(input_path, loader_module="test_model_export", code_path=[__file__],
                              data_path=model_pkl)
            output_path = tmp.path("output_model")
            result = CliRunner(
                env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(cli.commands,
                                                                             ['export', '-m',
                                                                              input_path, '-o',
                                                                              output_path])
            if result.exit_code:
                print('non-zero return code, output:', result.output, result.exception,
                      result.exc_info)
            self.assertEqual(0, result.exit_code)
            os.chdir(output_path)
            import sys
            sys.path.insert(0, '')
            print(sys.path)
            score = importlib.import_module("score")
            score.init()
            for i in range(0, len(self._linear_lr_predict)):
                json = '[{"col1":%f, "col2":%f}]' % tuple(self._X[i, :])
                x = score.run(json)
                self.assertEqual(self._linear_lr_predict[i], x[0])
        print("current dir", os.getcwd())
        assert os.path.exists(os.getcwd())


if __name__ == '__main__':
    unittest.main()
