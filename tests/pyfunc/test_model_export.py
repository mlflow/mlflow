from __future__ import print_function

import os
import pickle
import shutil
import tempfile
import unittest

from click.testing import CliRunner
import numpy as np
import pandas
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

from mlflow import pyfunc
import mlflow.pyfunc.cli
from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = sklearn.datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._knn = sklearn.neighbors.KNeighborsClassifier()
        self._knn.fit(self._X, self._y)
        self._knn_predict = self._knn.predict(self._X)
        self._linear_lr = sklearn.linear_model.LogisticRegression()
        self._linear_lr.fit(self._X, self._y)
        self._linear_lr_predict = self._linear_lr.predict(self._X)

    def test_model_save_load(self):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")
            m = Model(run_id="test", artifact_path="testtest")
            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              model=m)
            m2 = Model.load(os.path.join(path, "MLmodel"))
            print("m1", m.__dict__)
            print("m2", m2.__dict__)
            assert m.__dict__ == m2.__dict__
            x = pyfunc.load_pyfunc(path)
            xpred = x.predict(self._X)
            np.testing.assert_array_equal(self._knn_predict, xpred)

    def test_model_log(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            model_path = tmp.path("linear.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._linear_lr, f)
            tracking_dir = os.path.abspath(tmp.path("mlruns"))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            tracking.start_run()
            try:
                pyfunc.log_model(artifact_path="linear",
                                 data_path=model_path,
                                 loader_module=os.path.basename(__file__)[:-3],
                                 code_path=[__file__])

                run_id = tracking.active_run().info.run_uuid
                path = tracking._get_model_log_dir("linear", run_id)
                m = Model.load(os.path.join(path, "MLmodel"))
                print(m.__dict__)
                x = pyfunc.load_pyfunc("linear", run_id=run_id)
                xpred = x.predict(self._X)
                np.testing.assert_array_equal(self._linear_lr_predict, xpred)
            finally:
                tracking.end_run()
                tracking.set_tracking_uri(None)
                # Remove the log directory in order to avoid adding new tests to pytest...
                shutil.rmtree(tracking_dir)

    def test_model_serve(self):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")
            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              )

    def test_cli_predict(self):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")
            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              )
            input_csv_path = tmp.path("input.csv")
            pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
            output_csv_path = tmp.path("output.csv")
            runner = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"})
            result = runner.invoke(mlflow.pyfunc.cli.commands,
                                   ['predict', '--model-path', path, '-i',
                                    input_csv_path, '-o', output_csv_path])
            print("result", result.output)
            print(result.exc_info)
            print(result.exception)
            assert result.exit_code == 0
            result_df = pandas.read_csv(output_csv_path, header=None)
            np.testing.assert_array_equal(result_df.values.transpose()[0],
                                          self._knn.predict(self._X))
