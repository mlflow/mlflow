import os
import pickle
import types

import cloudpickle
import numpy as np
import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors
import yaml

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.model
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import _assert_pip_requirements


def _load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


@pytest.fixture
def pyfunc_custom_env_file(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_pip_deps=[
            "scikit-learn",
            "pytest",
            "cloudpickle",
            "-e " + os.path.dirname(mlflow.__path__[0]),
        ],
    )
    return conda_env


@pytest.fixture
def pyfunc_custom_env_dict():
    return _mlflow_conda_env(
        additional_pip_deps=[
            "scikit-learn",
            "pytest",
            "cloudpickle",
            "-e " + os.path.dirname(mlflow.__path__[0]),
        ],
    )


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


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


def test_model_save_load(sklearn_knn_model, iris_data, tmp_path, model_path):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=sk_model_path,
        loader_module=__name__,
        code_paths=[__file__],
        mlflow_model=model_config,
    )

    reloaded_model_config = Model.load(os.path.join(model_path, "MLmodel"))
    assert model_config.__dict__ == reloaded_model_config.__dict__
    assert mlflow.pyfunc.FLAVOR_NAME in reloaded_model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in reloaded_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )


def test_signature_and_examples_are_saved_correctly(sklearn_knn_model, iris_data):
    data = iris_data
    signature_ = infer_signature(*data)
    example_ = data[0][:3]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                with open(tmp.path("skmodel"), "wb") as f:
                    pickle.dump(sklearn_knn_model, f)
                path = tmp.path("model")
                mlflow.pyfunc.save_model(
                    path=path,
                    data_path=tmp.path("skmodel"),
                    loader_module=__name__,
                    code_paths=[__file__],
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_array_equal(_read_example(mlflow_model, path), example)


def test_model_log_load(sklearn_knn_model, iris_data, tmp_path):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=__name__,
            code_paths=[__file__],
        )
        pyfunc_model_path = _download_artifact_from_uri(
            f"runs:/{mlflow.active_run().info.run_id}/{pyfunc_artifact_path}"
        )

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_model(pyfunc_model_path)
    assert model_config.to_yaml() == reloaded_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )


def test_model_log_load_no_active_run(sklearn_knn_model, iris_data, tmp_path):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    assert mlflow.active_run() is None
    mlflow.pyfunc.log_model(
        artifact_path=pyfunc_artifact_path,
        data_path=sk_model_path,
        loader_module=__name__,
        code_paths=[__file__],
    )
    pyfunc_model_path = _download_artifact_from_uri(
        f"runs:/{mlflow.active_run().info.run_id}/{pyfunc_artifact_path}"
    )

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_model(pyfunc_model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )
    mlflow.end_run()


def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.save_model(path=model_path, data_path="/path/to/data")


def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model", data_path="/path/to/data")


def test_log_model_persists_specified_conda_env_file_in_mlflow_model_directory(
    sklearn_knn_model, tmp_path, pyfunc_custom_env_file
):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=__name__,
            code_paths=[__file__],
            conda_env=pyfunc_custom_env_file,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(f"runs:/{run_id}/{pyfunc_artifact_path}")

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env_file

    with open(pyfunc_custom_env_file) as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


def test_log_model_persists_specified_conda_env_dict_in_mlflow_model_directory(
    sklearn_knn_model, tmp_path, pyfunc_custom_env_dict
):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=__name__,
            code_paths=[__file__],
            conda_env=pyfunc_custom_env_dict,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(f"runs:/{run_id}/{pyfunc_artifact_path}")

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_dict


def test_log_model_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, tmp_path, pyfunc_custom_env_dict
):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=__name__,
            code_paths=[__file__],
            conda_env=pyfunc_custom_env_dict,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(f"runs:/{run_id}/{pyfunc_artifact_path}")

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    assert os.path.exists(saved_pip_req_path)

    with open(saved_pip_req_path) as f:
        requirements = f.read().split("\n")

    assert pyfunc_custom_env_dict["dependencies"][-1]["pip"] == requirements


def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model, tmp_path
):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=__name__,
            code_paths=[__file__],
        )
        model_uri = mlflow.get_artifact_uri(pyfunc_artifact_path)
    _assert_pip_requirements(model_uri, mlflow.pyfunc.get_default_pip_requirements())


def test_streamable_model_save_load(tmp_path, model_path):
    class StreamableModel:
        def __init__(self):
            pass

        def predict(self, model_input, params=None):
            pass

        def predict_stream(self, model_input, params=None):
            yield "test1"
            yield "test2"

    custom_model = StreamableModel()

    custom_model_path = os.path.join(tmp_path, "model.pkl")
    with open(custom_model_path, "wb") as f:
        cloudpickle.dump(custom_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=custom_model_path,
        loader_module=__name__,
        code_paths=[__file__],
        mlflow_model=model_config,
        streamable=True,
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_path)

    stream_result = loaded_pyfunc_model.predict_stream("single-input")
    assert isinstance(stream_result, types.GeneratorType)

    assert list(stream_result) == ["test1", "test2"]
