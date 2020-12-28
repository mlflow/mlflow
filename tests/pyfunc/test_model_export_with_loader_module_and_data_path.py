import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pytest
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
from mlflow.types import Schema, ColSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration


def _load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")  # pylint: disable=unexpected-keyword-arg


@pytest.fixture
def pyfunc_custom_env_file(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_conda_deps=["scikit-learn", "pytest", "cloudpickle"],
        additional_pip_deps=["-e " + os.path.dirname(mlflow.__path__[0])],
    )
    return conda_env


@pytest.fixture
def pyfunc_custom_env_dict():
    return _mlflow_conda_env(
        additional_conda_deps=["scikit-learn", "pytest", "cloudpickle"],
        additional_pip_deps=["-e " + os.path.dirname(mlflow.__path__[0])],
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
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_model_save_load(sklearn_knn_model, iris_data, tmpdir, model_path):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=sk_model_path,
        loader_module=os.path.basename(__file__)[:-3],
        code_path=[__file__],
        mlflow_model=model_config,
    )

    reloaded_model_config = Model.load(os.path.join(model_path, "MLmodel"))
    assert model_config.__dict__ == reloaded_model_config.__dict__
    assert mlflow.pyfunc.FLAVOR_NAME in reloaded_model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in reloaded_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(sklearn_knn_model, iris_data):
    data = iris_data
    signature_ = infer_signature(*data)
    example_ = data[0][
        :3,
    ]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                with open(tmp.path("skmodel"), "wb") as f:
                    pickle.dump(sklearn_knn_model, f)
                path = tmp.path("model")
                mlflow.pyfunc.save_model(
                    path=path,
                    data_path=tmp.path("skmodel"),
                    loader_module=os.path.basename(__file__)[:-3],
                    code_path=[__file__],
                    signature=signature,
                    input_example=example,
                )
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
    input_schema = Schema(
        [
            ColSpec("integer", "a"),
            ColSpec("long", "b"),
            ColSpec("float", "c"),
            ColSpec("double", "d"),
            ColSpec("boolean", "e"),
            ColSpec("string", "g"),
            ColSpec("binary", "f"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(
        data=[[1, 2, 3, 4, True, "x", bytes([1])]],
        columns=["b", "d", "a", "c", "e", "g", "f"],
        dtype=np.object,
    )
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.int64)
    pdf["c"] = pdf["c"].astype(np.float32)
    pdf["d"] = pdf["d"].astype(np.float64)
    # test that missing column raises
    with pytest.raises(MlflowException) as ex:
        res = pyfunc_model.predict(pdf[["b", "d", "a", "e", "g", "f"]])
    assert "Model input is missing columns" in str(ex)

    # test that extra column is ignored
    pdf["x"] = 1

    # test that columns are reordered, extra column is ignored
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.column_names()]).all())

    expected_types = dict(zip(input_schema.column_names(), input_schema.pandas_types()))
    actual_types = res.dtypes.to_dict()
    assert expected_types == actual_types

    # Test conversions
    # 1. long -> integer raises
    pdf["a"] = pdf["a"].astype(np.int64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["a"] = pdf["a"].astype(np.int32)
    # 2. integer -> long works
    pdf["b"] = pdf["b"].astype(np.int32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.column_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 3. unsigned int -> long works
    pdf["b"] = pdf["b"].astype(np.uint32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.column_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 4. unsigned int -> int raises
    pdf["a"] = pdf["a"].astype(np.uint32)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["a"] = pdf["a"].astype(np.int32)

    # 5. double -> float raises
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 6. float -> double works, double -> float does not
    pdf["d"] = pdf["d"].astype(np.float32)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types
    assert "Incompatible input types" in str(ex)
    pdf["d"] = pdf["d"].astype(np.float64)
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 7. int -> float raises
    pdf["c"] = pdf["c"].astype(np.int32)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 8. int -> double works
    pdf["d"] = pdf["d"].astype(np.int32)
    pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.column_names()]).all())
    assert res.dtypes.to_dict() == expected_types

    # 9. long -> double raises
    pdf["d"] = pdf["d"].astype(np.int64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["d"] = pdf["d"].astype(np.float64)

    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float32)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    pdf["b"] = pdf["b"].astype(np.int64)

    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    pdf["b"] = pdf["b"].astype(np.int64)
    assert "Incompatible input types" in str(ex)

    # 11. objects work
    pdf["b"] = pdf["b"].astype(np.object)
    pdf["d"] = pdf["d"].astype(np.object)
    pdf["e"] = pdf["e"].astype(np.object)
    pdf["f"] = pdf["f"].astype(np.object)
    pdf["g"] = pdf["g"].astype(np.object)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types

    # 8. np.ndarrays can be converted to dataframe but have no columns
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf.values)
    assert "Model input is missing columns" in str(ex)

    # 9. dictionaries of str -> list/nparray work
    arr = np.array([1, 2, 3])
    d = {
        "a": arr.astype("int32"),
        "b": arr.astype("int64"),
        "c": arr.astype("float32"),
        "d": arr.astype("float64"),
        "e": [True, False, True],
        "g": ["a", "b", "c"],
        "f": [bytes(0), bytes(1), bytes(1)],
    }
    res = pyfunc_model.predict(d)
    assert res.dtypes.to_dict() == expected_types

    # 10. dictionaries of str -> list[list] fail
    d = {
        "a": [arr.astype("int32")],
        "b": [arr.astype("int64")],
        "c": [arr.astype("float32")],
        "d": [arr.astype("float64")],
        "e": [[True, False, True]],
        "g": [["a", "b", "c"]],
        "f": [[bytes(0), bytes(1), bytes(1)]],
    }
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(d)
    assert "Incompatible input types" in str(ex)

    # 11. conversion to dataframe fails
    d = {
        "a": [1],
        "b": [1, 2],
        "c": [1, 2, 3],
    }
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(d)
    assert "This model contains a model signature, which suggests a DataFrame input." in str(ex)


def test_missing_value_hint_is_displayed_when_it_should():
    class TestModel(object):
        @staticmethod
        def predict(pdf):
            return pdf

    m = Model()
    input_schema = Schema([ColSpec("integer", "a")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[1], [None]], columns=["a"],)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    hint = "Hint: the type mismatch is likely caused by missing values."
    assert "Incompatible input types" in str(ex.value.message)
    assert hint in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1.5], [None]], columns=["a"],)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex)
    assert hint not in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1], [2]], columns=["a"], dtype=np.float64)
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(pdf)
    assert "Incompatible input types" in str(ex.value.message)
    assert hint not in str(ex.value.message)


def test_schema_enforcement_no_col_names():
    class TestModel(object):
        @staticmethod
        def predict(pdf):
            return pdf

    m = Model()
    input_schema = Schema([ColSpec("double"), ColSpec("double"), ColSpec("double")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    test_data = [[1.0, 2.0, 3.0]]

    # Can call with just a list
    assert pyfunc_model.predict(test_data).equals(pd.DataFrame(test_data))

    # Or can call with a DataFrame without column names
    assert pyfunc_model.predict(pd.DataFrame(test_data)).equals(pd.DataFrame(test_data))

    # # Or can call with a np.ndarray
    assert pyfunc_model.predict(pd.DataFrame(test_data).values).equals(pd.DataFrame(test_data))

    # Or with column names!
    pdf = pd.DataFrame(data=test_data, columns=["a", "b", "c"])
    assert pyfunc_model.predict(pdf).equals(pdf)

    # Must provide the right number of arguments
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict([[1.0, 2.0]])
    assert "the provided input only has 2 columns." in str(ex)

    # Must provide the right types
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict([[1, 2, 3]])
    assert "Can not safely convert int64 to float64" in str(ex)

    # Can only provide data type that can be converted to dataframe...
    with pytest.raises(MlflowException) as ex:
        pyfunc_model.predict(set([1, 2, 3]))
    assert "Expected input to be DataFrame or list. Found: set" in str(ex)

    # 9. dictionaries of str -> list/nparray work
    d = {"a": [1.0], "b": [2.0], "c": [3.0]}
    assert pyfunc_model.predict(d).equals(pd.DataFrame(d))


@pytest.mark.large
def test_model_log_load(sklearn_knn_model, iris_data, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=os.path.basename(__file__)[:-3],
            code_path=[__file__],
        )
        pyfunc_model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
            )
        )

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(pyfunc_model_path)
    assert model_config.to_yaml() == reloaded_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )


@pytest.mark.large
def test_model_log_load_no_active_run(sklearn_knn_model, iris_data, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    assert mlflow.active_run() is None
    mlflow.pyfunc.log_model(
        artifact_path=pyfunc_artifact_path,
        data_path=sk_model_path,
        loader_module=os.path.basename(__file__)[:-3],
        code_path=[__file__],
    )
    pyfunc_model_path = _download_artifact_from_uri(
        "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
        )
    )

    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(pyfunc_model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )
    mlflow.end_run()


@pytest.mark.large
def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(path=model_path, data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)


@pytest.mark.large
def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model", data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)


@pytest.mark.large
def test_log_model_persists_specified_conda_env_file_in_mlflow_model_directory(
    sklearn_knn_model, tmpdir, pyfunc_custom_env_file
):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=os.path.basename(__file__)[:-3],
            code_path=[__file__],
            conda_env=pyfunc_custom_env_file,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(
        "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=pyfunc_artifact_path)
    )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
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
    sklearn_knn_model, tmpdir, pyfunc_custom_env_dict
):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=os.path.basename(__file__)[:-3],
            code_path=[__file__],
            conda_env=pyfunc_custom_env_dict,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(
        "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=pyfunc_artifact_path)
    )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_dict


@pytest.mark.large
def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model, tmpdir
):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            data_path=sk_model_path,
            loader_module=os.path.basename(__file__)[:-3],
            code_path=[__file__],
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(
        "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=pyfunc_artifact_path)
    )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pyfunc.model.get_default_conda_env()
