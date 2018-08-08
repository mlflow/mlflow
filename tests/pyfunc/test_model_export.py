from __future__ import print_function

import os
import time
import six
import pickle
import shutil
import tempfile
import unittest

from click.testing import CliRunner
import psutil
import numpy as np
import pandas
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

import mlflow
from mlflow import pyfunc
import mlflow.pyfunc.cli
from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir
from multiprocessing import Process
import requests


def load_pyfunc(path):
    with open(path, "rb") as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')  # pylint: disable=unexpected-keyword-arg


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
            assert pyfunc.FLAVOR_NAME in m2.flavors
            assert pyfunc.PY_VERSION in m2.flavors[pyfunc.FLAVOR_NAME]
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
                assert pyfunc.FLAVOR_NAME in m.flavors
                assert pyfunc.PY_VERSION in m.flavors[pyfunc.FLAVOR_NAME]
                x = pyfunc.load_pyfunc("linear", run_id=run_id)
                xpred = x.predict(self._X)
                np.testing.assert_array_equal(self._linear_lr_predict, xpred)
            finally:
                tracking.end_run()
                tracking.set_tracking_uri(None)
                # Remove the log directory in order to avoid adding new tests to pytest...
                shutil.rmtree(tracking_dir)

    def _model_serve_with_conda_env(self, extra_args):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")

            # create a conda yaml that installs mlflow from source in-place mode
            conda_env_path = tmp.path("conda.yml")
            with open(conda_env_path, "w") as f:
                f.write("""
                        name: mlflow
                        channels:
                          - defaults
                        dependencies:
                          - pip:
                            - -e {}
                        """.format(os.path.abspath(os.path.join(mlflow.__path__[0], '..'))))

            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              conda_env=conda_env_path
                              )
            input_csv_path = tmp.path("input.csv")
            pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
            runner = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"})
            port = 5000

            def runserver():
                result = runner.invoke(mlflow.pyfunc.cli.commands,
                                       ['serve', '--model-path', path, '--port', port] + extra_args)
                print(result.exc_info)
                if result.exit_code != 0:
                    return -1

            process = Process(target=runserver)
            process.start()
            success = False
            while not success:
                try:
                    response = requests.post("http://localhost:{}/invocations".format(port),
                                             data=open(input_csv_path, 'rb'),
                                             headers={'Content-type': 'text/csv'})
                    response.close()
                    success = True
                except requests.ConnectionError:
                    pass
                time.sleep(5)

            assert process.is_alive(), "rest server died"
            for p in psutil.Process(process.pid).children(recursive=True):
                p.kill()
            process.terminate()

            process.join()
            assert not process.is_alive(), "server still alive after termination"
            if not success:
                raise RuntimeError("Fail to connect to the server")
            else:
                result_df = pandas.read_json(response.content)
                np.testing.assert_array_equal(result_df.values.transpose()[0],
                                              self._knn.predict(self._X))

    def test_model_serve_without_no_conda(self):
        self._model_serve_with_conda_env(extra_args=[])

    def test_model_serve_with_no_conda(self):
        self._model_serve_with_conda_env(extra_args=['--no-conda'])

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

    def _cli_predict_with_conda_env(self, extra_args):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)

            # create a conda yaml that installs mlflow from source in-place mode
            conda_env_path = tmp.path("conda.yml")
            with open(conda_env_path, "w") as f:
                f.write("""
                        name: mlflow
                        channels:
                          - defaults
                        dependencies:
                          - pip:
                            - -e {}
                        """.format(os.path.abspath(os.path.join(mlflow.__path__[0], '..'))))

            path = tmp.path("knn")
            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              conda_env=conda_env_path
                              )
            input_csv_path = tmp.path("input.csv")
            pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
            output_csv_path = tmp.path("output.csv")
            runner = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"})
            result = runner.invoke(mlflow.pyfunc.cli.commands,
                                   ['predict', '--model-path', path, '-i',
                                    input_csv_path, '-o', output_csv_path] + extra_args)
            print("result", result.output)
            print(result.exc_info)
            print(result.exception)
            assert result.exit_code == 0
            result_df = pandas.read_csv(output_csv_path, header=None)
            np.testing.assert_array_equal(result_df.values.transpose()[0],
                                          self._knn.predict(self._X))

    def test_cli_predict_without_no_conda(self):
        """Run prediction in MLModel specified conda env"""
        self._cli_predict_with_conda_env([])

    def test_cli_predict_with_no_conda(self):
        """Run prediction in current conda env"""
        self._cli_predict_with_conda_env(['--no-conda'])
