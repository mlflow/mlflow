import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pytest
import six
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
import mlflow.pyfunc.model
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.models.utils import _read_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import Schema, ColSpec, DataType
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration


def _load_pyfunc(path):
    with open(path, "rb") as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')  # pylint: disable=unexpected-keyword-arg


@pytest.fixture
def pyfunc_custom_env_file(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_conda_deps=["scikit-learn", "pytest", "cloudpickle"],
        additional_pip_deps=["-e " + os.path.dirname(mlflow.__path__[0])])
    return conda_env


@pytest.fixture
def pyfunc_custom_env_dict():
    return _mlflow_conda_env(
        additional_conda_deps=["scikit-learn", "pytest", "cloudpickle"],
        additional_pip_deps=["-e " + os.path.dirname(mlflow.__path__[0])])


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
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_model_save_load(sklearn_knn_model, iris_data, tmpdir, model_path):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(path=model_path,
                             data_path=sk_model_path,
                             loader_module=os.path.basename(__file__)[:-3],
                             code_path=[__file__],
                             mlflow_model=model_config)

    reloaded_model_config = Model.load(os.path.join(model_path, "MLmodel"))
    assert model_config.__dict__ == reloaded_model_config.__dict__
    assert mlflow.pyfunc.FLAVOR_NAME in reloaded_model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in reloaded_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0]))


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(sklearn_knn_model, iris_data):
    data = iris_data
    signature_ = infer_signature(*data)
    example_ = data[0][:3, ]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                with open(tmp.path("skmodel"), "wb") as f:
                    pickle.dump(sklearn_knn_model, f)
                path = tmp.path("model")
                mlflow.pyfunc.save_model(path=path,
                                         data_path=tmp.path("skmodel"),
                                         loader_module=os.path.basename(__file__)[:-3],
                                         code_path=[__file__],
                                         signature=signature,
                                         input_example=example)
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_schema_enforcement():
    class TestModel(object):
        @staticmethod
        def predict(pdf):
            return pdf
    m = Model()
    m.signature = ModelSignature(
        inputs=Schema([
            ColSpec("integer", "a"),
            ColSpec("double", "b"),
            ColSpec("string", "c"),
            ColSpec("integer", "d")
        ])
    )
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[1, "NaN", 2, {}, "3"]], columns=["b", "d", "a", "e", "c"])
    res = pyfunc_model.predict(pdf, 'NONE')
    assert all(res.columns == pdf.columns)

    assert (res.dtypes == pdf.dtypes).all()
    assert all((res == pdf).all())
    res1 = pyfunc_model.predict(pdf, 'LOOSE')
    assert all(res1.columns == ["a", "b", "c", "d", "e"])

    assert (res1[["a", "b", "c"]].dtypes.to_dict() == {
        "a": DataType.integer.to_pandas(),
        "b": DataType.double.to_pandas(),
        "c": DataType.string.to_pandas()
    })

    assert res.dtypes["d"] == pdf.dtypes["d"]
    assert res.dtypes["e"] == pdf.dtypes["e"]

    res1 = pyfunc_model.predict(pdf, 'LOOSE')
    assert all(res1.columns == ["a", "b", "c", "d", "e"])
    assert res1[["a", "c"]].dtypes.to_dict() == {
        "a": DataType.integer.to_pandas(),
        "c": DataType.string.to_pandas()
    }
    assert res.dtypes["d"] == pdf.dtypes["d"]
    assert res.dtypes["e"] == pdf.dtypes["e"]

    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf, 'STRICT') # extra column
    print(ex.value)
    print(dir(ex))
    assert "Model input does not match the expected schema." in ex.value.message
    with pytest.raises(MlflowException) as ex2:
        pyfunc_model.predict(pdf[["d", "a", "c"]], 'STRICT')  # missing column
    print(ex2)
    assert "Model input does not match the expected schema." in ex2.value.message
    with pytest.raises(MlflowException) as ex3:
        # type mismatch
        pyfunc_model.predict(pdf[["b", "d", "a", "c"]], 'STRICT')
    print(ex3)
    assert "Failed to convert column d" in ex3.value.message
    pyfunc_model._model_meta.signature = ModelSignature(
        inputs=Schema([
            ColSpec("integer", "a"),
            ColSpec("double", "b"),
            ColSpec("string", "c"),
        ])
    )
    res2 = pyfunc_model.predict(pdf[["b", "a", "c"]], 'STRICT')
    assert all(res2.columns == ["a", "b", "c"])
    assert res2[["a", "b", "c"]].dtypes.to_dict() == {
        "a": DataType.integer.to_pandas(),
        "b": DataType.double.to_pandas(),
        "c": DataType.string.to_pandas()
    }








@pytest.mark.large
def test_model_log_load(sklearn_knn_model, iris_data, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                data_path=sk_model_path,
                                loader_module=os.path.basename(__file__)[:-3],
                                code_path=[__file__])
        pyfunc_model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path))

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(pyfunc_model_path)
    assert model_config.to_yaml() == reloaded_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0]))


@pytest.mark.large
def test_model_log_load_no_active_run(sklearn_knn_model, iris_data, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    assert mlflow.active_run() is None
    mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                            data_path=sk_model_path,
                            loader_module=os.path.basename(__file__)[:-3],
                            code_path=[__file__])
    pyfunc_model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
        run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path))

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(pyfunc_model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0]))
    mlflow.end_run()


@pytest.mark.large
def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(path=model_path,
                                 data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)


@pytest.mark.large
def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model",
                                data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)


@pytest.mark.large
def test_log_model_persists_specified_conda_env_file_in_mlflow_model_directory(
        sklearn_knn_model, tmpdir, pyfunc_custom_env_file):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                data_path=sk_model_path,
                                loader_module=os.path.basename(__file__)[:-3],
                                code_path=[__file__],
                                conda_env=pyfunc_custom_env_file)
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
        run_id=run_id, artifact_path=pyfunc_artifact_path))

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env_file

    with open(pyfunc_custom_env_file, "r") as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


@pytest.mark.large
def test_log_model_persists_specified_conda_env_dict_in_mlflow_model_directory(
        sklearn_knn_model, tmpdir, pyfunc_custom_env_dict):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                data_path=sk_model_path,
                                loader_module=os.path.basename(__file__)[:-3],
                                code_path=[__file__],
                                conda_env=pyfunc_custom_env_dict)
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
        run_id=run_id, artifact_path=pyfunc_artifact_path))

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_dict


@pytest.mark.large
def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        sklearn_knn_model, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                data_path=sk_model_path,
                                loader_module=os.path.basename(__file__)[:-3],
                                code_path=[__file__])
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
        run_id=run_id, artifact_path=pyfunc_artifact_path))

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pyfunc.model.get_default_conda_env()
