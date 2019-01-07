from __future__ import print_function

import os
import time
import six
import pickle
import shutil
import tempfile
import unittest
import requests
import signal
from subprocess import Popen, STDOUT

from click.testing import CliRunner
import numpy as np
import pandas
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

import mlflow
from mlflow import pyfunc
from mlflow import tracking
import mlflow.pyfunc.cli
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir


def _load_pyfunc(path):
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
            mlflow.set_tracking_uri("file://%s" % tracking_dir)
            mlflow.start_run()
            try:
                pyfunc.log_model(artifact_path="linear",
                                 data_path=model_path,
                                 loader_module=os.path.basename(__file__)[:-3],
                                 code_path=[__file__])

                run_id = mlflow.active_run().info.run_uuid
                path = tracking.utils._get_model_log_dir("linear", run_id)
                m = Model.load(os.path.join(path, "MLmodel"))
                print(m.__dict__)
                assert pyfunc.FLAVOR_NAME in m.flavors
                assert pyfunc.PY_VERSION in m.flavors[pyfunc.FLAVOR_NAME]
                x = pyfunc.load_pyfunc("linear", run_id=run_id)
                xpred = x.predict(self._X)
                np.testing.assert_array_equal(self._linear_lr_predict, xpred)
            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(None)
                # Remove the log directory in order to avoid adding new tests to pytest...
                shutil.rmtree(tracking_dir)

    def test_add_to_model_adds_specified_kwargs_to_mlmodel_configuration(self):
        custom_kwargs = {
            "key1": "value1",
            "key2": 20,
            "key3": range(10),
        }
        model_config = Model()
        pyfunc.add_to_model(model=model_config,
                            loader_module=os.path.basename(__file__)[:-3],
                            data="data",
                            code="code",
                            env=None,
                            **custom_kwargs)

        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert all([item in model_config.flavors[pyfunc.FLAVOR_NAME] for item in custom_kwargs])

    def _create_conda_env_file(self, tmp):
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
        return conda_env_path

    def _model_serve_with_conda_env(self, extra_args):
        with TempDir() as tmp:
            model_path = tmp.path("knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self._knn, f)
            path = tmp.path("knn")

            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              conda_env=self._create_conda_env_file(tmp)
                              )
            input_csv_path = tmp.path("input.csv")
            pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
            port = 5000
            process = Popen(['mlflow', 'pyfunc', 'serve',
                             '--model-path', path, '--port', str(port)] + extra_args,
                            stderr=STDOUT,
                            preexec_fn=os.setsid)
            time.sleep(5)
            try:
                assert process.poll() is None, "server died prematurely"
                success = False
                failcount = 0
                while not success and failcount < 3 and process.poll() is None:
                    try:
                        response = requests.post("http://localhost:{}/invocations".format(port),
                                                 data=open(input_csv_path, 'rb'),
                                                 headers={'Content-type': 'text/csv'})
                        response.close()
                        success = True
                    except requests.ConnectionError:
                        time.sleep(5)
                        failcount += 1
            finally:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kill process + children
                time.sleep(0.5)
                assert process.poll() is not None, "server not dead"

            # check result
            if not success:
                raise RuntimeError("Fail to connect to the server")
            else:
                result_df = pandas.read_json(response.content)
                np.testing.assert_array_equal(result_df.values.transpose()[0],
                                              self._knn.predict(self._X))

    def test_model_serve_with_conda(self):
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
            input_csv_path = tmp.path("input with spaces.csv")
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
            path = tmp.path("knn")
            pyfunc.save_model(dst_path=path,
                              data_path=model_path,
                              loader_module=os.path.basename(__file__)[:-3],
                              code_path=[__file__],
                              conda_env=self._create_conda_env_file(tmp)
                              )
            input_csv_path = tmp.path("input with spaces.csv")
            pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
            output_csv_path = tmp.path("output.csv")
            process = Popen(['mlflow', 'pyfunc', 'predict',
                             '--model-path', path,
                             '-i', input_csv_path,
                             '-o', output_csv_path] + extra_args,
                            stderr=STDOUT,
                            preexec_fn=os.setsid)
            process.wait()
            result_df = pandas.read_csv(output_csv_path, header=None)
            np.testing.assert_array_equal(result_df.values.transpose()[0],
                                          self._knn.predict(self._X))

    def test_cli_predict_with_conda(self):
        """Run prediction in MLModel specified conda env"""
        self._cli_predict_with_conda_env([])

    def test_cli_predict_with_no_conda(self):
        """Run prediction in current conda env"""
        self._cli_predict_with_conda_env(['--no-conda'])
