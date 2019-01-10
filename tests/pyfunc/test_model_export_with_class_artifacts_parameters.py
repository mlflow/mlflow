from __future__ import print_function

import mock
import os
import json
import sys
from distutils.version import StrictVersion
from subprocess import Popen, STDOUT

import numpy as np
import pandas as pd
import pandas.testing
import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors
import yaml

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.cli
import mlflow.pyfunc.model
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking.utils import get_artifact_uri as utils_get_artifact_uri
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

import tests
from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import score_model_in_sagemaker_docker_container


def get_model_class():
    """
    Defines a custom Python model class that wraps a scikit-learn estimator.
    This can be invoked within a pytest fixture to define the class in the ``__main__`` scope.
    Alternatively, it can be invoked within a module to define the class in the module's scope.
    """
    class CustomSklearnModel(mlflow.pyfunc.PythonModel):

        def __init__(self, context):
            super(CustomSklearnModel, self).__init__(context)
            self.model = mlflow.sklearn.load_model(path=context.artifacts["sk_model"])

        def predict(self, model_input):
            return self.context.parameters["predict_fn"](self.model, model_input)

    return CustomSklearnModel


class ModuleScopedSklearnModel(get_model_class()):
    """
    A custom Python model class defined in the test module scope.
    """
    pass


@pytest.fixture(scope="module")
def main_scoped_model_class():
    """
    A custom Python model class defined in the ``__main__`` scope.
    """
    return get_model_class()


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


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


@pytest.fixture
def pyfunc_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
            conda_env,
            additional_conda_deps=["scikit-learn", "pytest", "cloudpickle"])
    return conda_env


def test_model_save_load(sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)
    np.testing.assert_array_equal(
            loaded_pyfunc_model.predict(model_input=iris_data[0]),
            test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]))


def test_model_log_load(sklearn_knn_model, main_scoped_model_class, iris_data):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_uuid

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_model": utils_get_artifact_uri(
                                        artifact_path=sklearn_artifact_path,
                                        run_id=sklearn_run_id)
                                },
                                parameters={
                                    "predict_fn": test_predict
                                },
                                model_class=main_scoped_model_class)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(path=pyfunc_artifact_path, run_id=pyfunc_run_id)
    np.testing.assert_array_equal(
            loaded_pyfunc_model.predict(model_input=iris_data[0]),
            test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]))


def test_add_to_model_adds_specified_kwargs_to_mlmodel_configuration():
    custom_kwargs = {
        "key1": "value1",
        "key2": 20,
        "key3": range(10),
    }
    model_config = Model()
    mlflow.pyfunc.add_to_model(model=model_config,
                               loader_module=os.path.basename(__file__)[:-3],
                               data="data",
                               code="code",
                               env=None,
                               **custom_kwargs)

    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert all([item in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME] for item in custom_kwargs])


def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_main_scoped_class(
        sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)
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


def test_pyfunc_model_serving_with_conda_env_activation_succeeds_with_main_scoped_class(
        sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)
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


def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_module_scoped_class(
        sklearn_knn_model, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=ModuleScopedSklearnModel,
                             code_path=[os.path.dirname(tests.__file__)])
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


def test_pyfunc_cli_predict_command_without_conda_env_activation_succeeds(
        sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)
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
        sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)
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


def test_save_model_specifying_model_dependency_with_different_major_python_verison_logs_warning(
        sklearn_knn_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    sk_model_config_path = os.path.join(sklearn_model_path, "MLmodel")
    sk_model_config = Model.load(sk_model_config_path)
    assert mlflow.pyfunc.FLAVOR_NAME in sk_model_config.flavors
    sk_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.PY_VERSION] = (
        "2.7.0" if sys.version_info >= (3, 0) else "3.6.0"
    )
    sk_model_config.save(sk_model_config_path)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text.format(*args, **kwargs))

    with mock.patch("mlflow.pyfunc._logger.warn") as warn_mock:
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "pyfunc_model"),
                                 artifacts={
                                    "sk_model": sklearn_model_path
                                 },
                                 parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                 },
                                 model_class=main_scoped_model_class)

    assert any([
        "MLflow model that was saved with a different major version of Python" in log_message
        for log_message in log_messages
    ])


def test_save_model_specifying_model_dependency_with_same_major_python_version_does_not_log_warning(
        sklearn_knn_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    sk_model_config_path = os.path.join(sklearn_model_path, "MLmodel")
    sk_model_config = Model.load(sk_model_config_path)
    sk_model_py_version = sk_model_config.flavors.get(mlflow.pyfunc.FLAVOR_NAME, {}).get(
        mlflow.pyfunc.PY_VERSION, None)
    assert sk_model_py_version is not None
    assert StrictVersion(sk_model_py_version).version[0] == sys.version_info.major

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text.format(*args, **kwargs))

    with mock.patch("mlflow.pyfunc._logger.warn") as warn_mock:
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "pyfunc_model"),
                                 artifacts={
                                    "sk_model": sklearn_model_path
                                 },
                                 parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                 },
                                 model_class=main_scoped_model_class)

    assert not any([
        "MLflow model that was saved with a different major version of Python" in log_message
        for log_message in log_messages
    ])


def test_save_model_specifying_model_dependency_with_different_cloudpickle_verison_logs_warning(
        sklearn_knn_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    with mock.patch("cloudpickle.__version__") as cloudpickle_version_mock:
        cloudpickle_version_mock.__str__ = lambda *args, **kwargs: "0.4.6"
        mlflow.sklearn.save_model(
            sk_model=sklearn_knn_model,
            path=sklearn_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text.format(*args, **kwargs))

    with mock.patch("mlflow.pyfunc._logger.warn") as warn_mock,\
            mock.patch("cloudpickle.__version__") as cloudpickle_version_mock:
        warn_mock.side_effect = custom_warn
        cloudpickle_version_mock.__str__ = lambda *args, **kwargs: "0.5.8"
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "pyfunc_model"),
                                 artifacts={
                                    "sk_model": sklearn_model_path
                                 },
                                 parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                 },
                                 model_class=main_scoped_model_class)

    assert any([
        "MLflow model that contains a dependency on either a different version or a"
        " range of versions of the CloudPickle library" in log_message
        for log_message in log_messages
    ])


def test_save_model_specifying_model_dependency_with_same_cloudpickle_verison_does_not_log_warning(
        sklearn_knn_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model,
                              path=sklearn_model_path,
                              serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text.format(*args, **kwargs))

    with mock.patch("mlflow.pyfunc._logger.warn") as warn_mock:
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "pyfunc_model"),
                                 artifacts={
                                    "sk_model": sklearn_model_path
                                 },
                                 parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                 },
                                 model_class=main_scoped_model_class)

    assert not any([
        "MLflow model that contains a dependency on either a different version or a"
        " range of versions of the CloudPickle library" in log_message
        for log_message in log_messages
    ])


def test_save_model_persists_specified_conda_env_in_mlflow_model_directory(
        sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model,
                              path=sklearn_model_path,
                              serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": lambda sk_model, model_input: None
                             },
                             model_class=main_scoped_model_class,
                             conda_env=pyfunc_custom_env)

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env, "r") as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


def test_log_model_persists_specified_conda_env_in_mlflow_model_directory(
        sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_uuid

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_model": utils_get_artifact_uri(
                                        artifact_path=sklearn_artifact_path,
                                        run_id=sklearn_run_id)
                                },
                                parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                },
                                model_class=main_scoped_model_class,
                                conda_env=pyfunc_custom_env)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    pyfunc_model_path = _get_model_log_dir(pyfunc_artifact_path, pyfunc_run_id)
    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env, "r") as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


def test_save_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        sklearn_logreg_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_logreg_model, path=sklearn_model_path)

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": lambda sk_model, model_input: None
                             },
                             model_class=main_scoped_model_class)

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pyfunc.model.DEFAULT_CONDA_ENV


def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        sklearn_knn_model, main_scoped_model_class):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_uuid

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_model": utils_get_artifact_uri(
                                        artifact_path=sklearn_artifact_path,
                                        run_id=sklearn_run_id)
                                },
                                parameters={
                                    "predict_fn": lambda sk_model, model_input: None
                                },
                                model_class=main_scoped_model_class)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    pyfunc_model_path = _get_model_log_dir(pyfunc_artifact_path, pyfunc_run_id)
    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pyfunc.model.DEFAULT_CONDA_ENV


def test_save_model_correctly_resolves_directory_artifact_with_nested_contents(
        tmpdir, model_path, iris_data):
    directory_artifact_path = os.path.join(str(tmpdir), "directory_artifact")
    nested_file_relative_path = os.path.join(
        "my", "somewhat", "heavily", "nested", "directory", "myfile.txt")
    nested_file_path = os.path.join(directory_artifact_path, nested_file_relative_path)
    os.makedirs(os.path.dirname(nested_file_path))
    nested_file_text = "some sample file text"
    with open(nested_file_path, "w") as f:
        f.write(nested_file_text)

    class ArtifactValidationModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            expected_file_path = os.path.join(
                self.context.artifacts["testdir"], nested_file_relative_path)
            if not os.path.exists(expected_file_path):
                return False
            else:
                with open(expected_file_path, "r") as f:
                    return (f.read() == nested_file_text)

    mlflow.pyfunc.save_model(dst_path=model_path,
                             artifacts={
                                 "testdir": directory_artifact_path
                             },
                             model_class=ArtifactValidationModel)

    loaded_model = mlflow.pyfunc.load_pyfunc(model_path)
    assert loaded_model.predict(iris_data[0])


def test_save_model_with_no_artifacts_or_parameters_does_not_produce_artifacts_or_parameters_dirs(
        tmpdir):
    no_params_or_artifacts_model_path = os.path.join(str(tmpdir), "no_params_or_artifacts")
    mlflow.pyfunc.save_model(dst_path=no_params_or_artifacts_model_path,
                             model_class=ModuleScopedSklearnModel,
                             parameters=None,
                             artifacts=None)

    no_artifacts_model_path = os.path.join(str(tmpdir), "no_artifacts")
    mlflow.pyfunc.save_model(dst_path=no_artifacts_model_path,
                             model_class=ModuleScopedSklearnModel,
                             parameters={
                                "sample_set": set(range(10)),
                             },
                             artifacts=None)

    no_params_model_path = os.path.join(str(tmpdir), "no_params")
    mlflow.pyfunc.save_model(dst_path=no_params_model_path,
                             model_class=ModuleScopedSklearnModel,
                             parameters=None,
                             artifacts={
                                "no_artifacts_model": no_artifacts_model_path,
                             })

    model_paths_and_expected_existence_results = {
        no_params_or_artifacts_model_path: {
            "artifacts_exists": False,
            "params_exists": False,
        },
        no_artifacts_model_path: {
            "artifacts_exists": False,
            "params_exists": True,
        },
        no_params_model_path: {
            "artifacts_exists": True,
            "params_exists": False,
        },
    }

    for model_path, expected_existence_result in\
            model_paths_and_expected_existence_results.items():
        assert os.path.exists(model_path)
        assert os.path.exists(os.path.join(model_path, "artifacts")) ==\
            expected_existence_result["artifacts_exists"]
        assert os.path.exists(os.path.join(model_path, "parameters")) ==\
            expected_existence_result["params_exists"]

        pyfunc_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
        assert (mlflow.pyfunc.model.CONFIG_KEY_ARTIFACTS in pyfunc_conf) ==\
            expected_existence_result["artifacts_exists"]
        assert (mlflow.pyfunc.model.CONFIG_KEY_PARAMETERS in pyfunc_conf) ==\
            expected_existence_result["params_exists"]


def test_save_model_with_model_class_parameter_of_invalid_type_raises_exeption(tmpdir):
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "model1"),
                                 model_class="not the right type")
    assert "`model_class` must be a class object" in str(exc_info)

    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=os.path.join(str(tmpdir), "model2"),
                                 model_class="not the right type")
    assert "`model_class` must be a class object" in str(exc_info)


def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=model_path,
                                 parameters={
                                    "param": range(10),
                                 },
                                 model_class=None)
    assert "`model_class` argument was not provided" in str(exc_info)

    model_class = ModuleScopedSklearnModel
    loader_module = __name__
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=model_path,
                                 model_class=model_class,
                                 loader_module=loader_module)
    assert "The following sets of arguments cannot be specified together" in str(exc_info)
    assert str(model_class) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=model_path,
                                 model_class=model_class,
                                 data_path="/path/to/data",
                                 artifacts={
                                    "artifact1": "/path/to/artifact",
                                 })
    assert "The following sets of arguments cannot be specified together" in str(exc_info)


def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model",
                                parameters={
                                    "param": range(10),
                                },
                                model_class=None)
    assert "`model_class` argument was not provided" in str(exc_info)

    model_class = ModuleScopedSklearnModel
    loader_module = __name__
    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model",
                                model_class=model_class,
                                loader_module=loader_module)
    assert "The following sets of arguments cannot be specified together" in str(exc_info)
    assert str(model_class) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model",
                                model_class=model_class,
                                data_path="/path/to/data",
                                artifacts={
                                    "artifact1": "/path/to/artifact",
                                })
    assert "The following sets of arguments cannot be specified together" in str(exc_info)


@pytest.mark.large
def test_sagemaker_docker_model_scoring_with_default_conda_env(
        sklearn_logreg_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_logreg_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(dst_path=pyfunc_model_path,
                             artifacts={
                                "sk_model": sklearn_model_path
                             },
                             parameters={
                                "predict_fn": test_predict
                             },
                             model_class=main_scoped_model_class)
    reloaded_pyfunc = mlflow.pyfunc.load_pyfunc(path=pyfunc_model_path)

    inference_df = pd.DataFrame(iris_data[0])
    scoring_response = score_model_in_sagemaker_docker_container(
            model_path=pyfunc_model_path,
            data=inference_df,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            flavor=mlflow.pyfunc.FLAVOR_NAME)
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    pandas.testing.assert_frame_equal(
        deployed_model_preds,
        pd.DataFrame(reloaded_pyfunc.predict(inference_df)),
        check_dtype=False,
        check_less_precise=6)
