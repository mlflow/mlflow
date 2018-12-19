from __future__ import print_function

import os
import json
from subprocess import Popen, STDOUT

import numpy as np
import pandas as pd
import pandas.testing
import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.cli
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import pyfunc_serve_and_score_model


@pytest.fixture(scope="module")
def model_class():
    class TestSklearnModel(mlflow.pyfunc.PythonModel):

        def __init__(self, context):
            super(TestSklearnModel, self).__init__(context)
            self.model = mlflow.sklearn.load_model(path=context.artifacts["sk_model"])

        def predict(self, input_df):
            return self.context.parameters["predict_fn"](self.model, input_df)
    
    return TestSklearnModel


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x,y


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_data):
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


@pytest.fixture(scope="module")
def sklearn_logreg_model(iris_data):
    x, y = iris_data
    linear_lr = sklearn.linear_model.LogisticRegression()
    linear_lr.fit(x, y)
    return linear_lr


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


def test_model_save_load(sklearn_knn_model, model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    
    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=model_class)

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)
    np.testing.assert_array_equal(
            loaded_pyfunc_model.predict(input_df=iris_data[0]),
            test_predict(sk_model=sklearn_knn_model, input_df=iris_data[0]))


def test_model_log_load(sklearn_knn_model, model_class, iris_data):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_uuid

    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_model": mlflow.get_artifact_uri(
                                        artifact_path=sklearn_artifact_path,
                                        run_id=sklearn_run_id)
                                },
                                parameters={
                                    "predict_fn": test_predict
                                },
                                model_class=model_class)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_artifact_path, run_id=pyfunc_run_id)
    np.testing.assert_array_equal(
            loaded_pyfunc_model.predict(input_df=iris_data[0]),
            test_predict(sk_model=sklearn_knn_model, input_df=iris_data[0]))

# class TestModelExport(unittest.TestCase):
#     def setUp(self):
#         self._tmp = tempfile.mkdtemp()
#         iris = sklearn.datasets.load_iris()
#         self._X = iris.data[:, :2]  # we only take the first two features.
#         self._y = iris.target
#         self._knn = sklearn.neighbors.KNeighborsClassifier()
#         self._knn.fit(self._X, self._y)
#         self._knn_predict = self._knn.predict(self._X)
#         self._linear_lr = sklearn.linear_model.LogisticRegression()
#         self._linear_lr.fit(self._X, self._y)
#         self._linear_lr_predict = self._linear_lr.predict(self._X)
#
#     def test_model_save_load(self):
#         with TempDir() as tmp:
#             model_path = tmp.path("knn.pkl")
#             with open(model_path, "wb") as f:
#                 pickle.dump(self._knn, f)
#             path = tmp.path("knn")
#             m = Model(run_id="test", artifact_path="testtest")
#             pyfunc.save_model(dst_path=path,
#                               data_path=model_path,
#                               loader_module=os.path.basename(__file__)[:-3],
#                               code_path=[__file__],
#                               model=m)
#             m2 = Model.load(os.path.join(path, "MLmodel"))
#             print("m1", m.__dict__)
#             print("m2", m2.__dict__)
#             assert m.__dict__ == m2.__dict__
#             assert pyfunc.FLAVOR_NAME in m2.flavors
#             assert pyfunc.PY_VERSION in m2.flavors[pyfunc.FLAVOR_NAME]
#             x = pyfunc.load_pyfunc(path)
#             xpred = x.predict(self._X)
#             np.testing.assert_array_equal(self._knn_predict, xpred)
#
#     def test_model_log(self):
#         with TempDir(chdr=True, remove_on_exit=True) as tmp:
#             model_path = tmp.path("linear.pkl")
#             with open(model_path, "wb") as f:
#                 pickle.dump(self._linear_lr, f)
#             tracking_dir = os.path.abspath(tmp.path("mlruns"))
#             mlflow.set_tracking_uri("file://%s" % tracking_dir)
#             mlflow.start_run()
#             try:
#                 pyfunc.log_model(artifact_path="linear",
#                                  data_path=model_path,
#                                  loader_module=os.path.basename(__file__)[:-3],
#                                  code_path=[__file__])
#
#                 run_id = mlflow.active_run().info.run_uuid
#                 path = tracking.utils._get_model_log_dir("linear", run_id)
#                 m = Model.load(os.path.join(path, "MLmodel"))
#                 print(m.__dict__)
#                 assert pyfunc.FLAVOR_NAME in m.flavors
#                 assert pyfunc.PY_VERSION in m.flavors[pyfunc.FLAVOR_NAME]
#                 x = pyfunc.load_pyfunc("linear", run_id=run_id)
#                 xpred = x.predict(self._X)
#                 np.testing.assert_array_equal(self._linear_lr_predict, xpred)
#             finally:
#                 mlflow.end_run()
#                 mlflow.set_tracking_uri(None)
#                 # Remove the log directory in order to avoid adding new tests to pytest...
#                 shutil.rmtree(tracking_dir)
#
#     def _create_conda_env_file(self, tmp):
#         conda_env_path = tmp.path("conda.yml")
#         with open(conda_env_path, "w") as f:
#             f.write("""
#                     name: mlflow
#                     channels:
#                       - defaults
#                     dependencies:
#                       - pip:
#                         - -e {}
#                     """.format(os.path.abspath(os.path.join(mlflow.__path__[0], '..'))))
#         return conda_env_path
#

def test_pyfunc_model_serving_without_conda_env_activation_succeeds(
        sklearn_knn_model, model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    
    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=model_class)
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)
    
    sample_input = pd.DataFrame(iris_data[0]) 
    scoring_response = pyfunc_serve_and_score_model(
            model_path=pyfunc_model_path, 
            data=sample_input, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            extra_args=["--no-conda"])
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)),
        loaded_pyfunc_model.predict(sample_input))


def test_pyfunc_model_serving_with_conda_env_activation_succeeds(
        sklearn_knn_model, model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    
    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=model_class)
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)
    
    sample_input = pd.DataFrame(iris_data[0]) 
    scoring_response = pyfunc_serve_and_score_model(
            model_path=pyfunc_model_path, 
            data=sample_input, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)),
        loaded_pyfunc_model.predict(sample_input))


def test_pyfunc_cli_predict_command_without_conda_env_activation_succeeds(
        sklearn_knn_model, model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    
    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=model_class)
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(str(tmpdir), "input with spaces.csv") 
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_csv_path = os.path.join(str(tmpdir), "output.csv")
    process = Popen(['mlflow', 'pyfunc', 'predict',
                     '--model-path', pyfunc_model_path,
                     '-i', input_csv_path,
                     '-o', output_csv_path,
                     '--no-conda'],
                    stderr=STDOUT,
                    preexec_fn=os.setsid)
    process.wait()
    result_df = pandas.read_csv(output_csv_path, header=None)
    np.testing.assert_array_equal(result_df.values.transpose()[0], 
                                  loaded_pyfunc_model.predict(sample_input))


def test_pyfunc_cli_predict_command_with_conda_env_activation_succeeds(
        sklearn_knn_model, model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    
    def test_predict(sk_model, input_df):
        return sk_model.predict(input_df) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=model_class)
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(str(tmpdir), "input with spaces.csv") 
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_csv_path = os.path.join(str(tmpdir), "output.csv")
    process = Popen(['mlflow', 'pyfunc', 'predict',
                     '--model-path', pyfunc_model_path,
                     '-i', input_csv_path,
                     '-o', output_csv_path],
                    stderr=STDOUT,
                    preexec_fn=os.setsid)
    process.wait()
    result_df = pandas.read_csv(output_csv_path, header=None)
    np.testing.assert_array_equal(result_df.values.transpose()[0], 
                                  loaded_pyfunc_model.predict(sample_input)) 
#
#     def _cli_predict_with_conda_env(self, extra_args):
#         with TempDir() as tmp:
#             model_path = tmp.path("knn.pkl")
#             with open(model_path, "wb") as f:
#                 pickle.dump(self._knn, f)
#
#             # create a conda yaml that installs mlflow from source in-place mode
#             path = tmp.path("knn")
#             pyfunc.save_model(dst_path=path,
#                               data_path=model_path,
#                               loader_module=os.path.basename(__file__)[:-3],
#                               code_path=[__file__],
#                               conda_env=self._create_conda_env_file(tmp)
#                               )
#             input_csv_path = tmp.path("input with spaces.csv")
#             pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
#             output_csv_path = tmp.path("output.csv")
#             process = Popen(['mlflow', 'pyfunc', 'predict',
#                              '--model-path', path,
#                              '-i', input_csv_path,
#                              '-o', output_csv_path] + extra_args,
#                             stderr=STDOUT,
#                             preexec_fn=os.setsid)
#             process.wait()
#             result_df = pandas.read_csv(output_csv_path, header=None)
#             np.testing.assert_array_equal(result_df.values.transpose()[0],
#                                           self._knn.predict(self._X))
#
#     def test_cli_predict_with_conda(self):
#         """Run prediction in MLModel specified conda env"""
#         self._cli_predict_with_conda_env([])
#
#     def test_cli_predict_with_no_conda(self):
#         """Run prediction in current conda env"""
#         self._cli_predict_with_conda_env(['--no-conda'])
