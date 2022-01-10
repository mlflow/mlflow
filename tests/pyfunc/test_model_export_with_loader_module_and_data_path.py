import os
import pickle
import yaml
import re

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
from mlflow.types import Schema, ColSpec, TensorSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.helper_functions import _assert_pip_requirements


class TestModel:
    @staticmethod
    def predict(pdf):
        return pdf


def _load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")  # pylint: disable=unexpected-keyword-arg


@pytest.fixture
def pyfunc_custom_env_file(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
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
        loader_module=__name__,
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
                    loader_module=__name__,
                    code_path=[__file__],
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert np.array_equal(_read_example(mlflow_model, path), example)


def test_column_schema_enforcement():
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
            ColSpec("datetime", "h"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(
        data=[[1, 2, 3, 4, True, "x", bytes([1]), "2021-01-01 00:00:00.1234567"]],
        columns=["b", "d", "a", "c", "e", "g", "f", "h"],
        dtype=np.object,
    )
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.int64)
    pdf["c"] = pdf["c"].astype(np.float32)
    pdf["d"] = pdf["d"].astype(np.float64)
    pdf["h"] = pdf["h"].astype(np.datetime64)
    # test that missing column raises
    match_missing_inputs = "Model is missing inputs"
    with pytest.raises(MlflowException, match=match_missing_inputs):
        res = pyfunc_model.predict(pdf[["b", "d", "a", "e", "g", "f", "h"]])

    # test that extra column is ignored
    pdf["x"] = 1

    # test that columns are reordered, extra column is ignored
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())

    expected_types = dict(zip(input_schema.input_names(), input_schema.pandas_types()))
    # MLflow datetime type in input_schema does not encode precision, so add it for assertions
    expected_types["h"] = np.dtype("datetime64[ns]")
    # np.object cannot be converted to pandas Strings at the moment
    expected_types["f"] = np.object
    expected_types["g"] = np.object
    actual_types = res.dtypes.to_dict()
    assert expected_types == actual_types

    # Test conversions
    # 1. long -> integer raises
    pdf["a"] = pdf["a"].astype(np.int64)
    match_incompatible_inputs = "Incompatible input types"
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)
    # 2. integer -> long works
    pdf["b"] = pdf["b"].astype(np.int32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 3. unsigned int -> long works
    pdf["b"] = pdf["b"].astype(np.uint32)
    res = pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types
    pdf["b"] = pdf["b"].astype(np.int64)

    # 4. unsigned int -> int raises
    pdf["a"] = pdf["a"].astype(np.uint32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)

    # 5. double -> float raises
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 6. float -> double works, double -> float does not
    pdf["d"] = pdf["d"].astype(np.float32)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types
    pdf["d"] = pdf["d"].astype(np.float64)
    pdf["c"] = pdf["c"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 7. int -> float raises
    pdf["c"] = pdf["c"].astype(np.int32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["c"] = pdf["c"].astype(np.float32)

    # 8. int -> double works
    pdf["d"] = pdf["d"].astype(np.int32)
    pyfunc_model.predict(pdf)
    assert all((res == pdf[input_schema.input_names()]).all())
    assert res.dtypes.to_dict() == expected_types

    # 9. long -> double raises
    pdf["d"] = pdf["d"].astype(np.int64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["d"] = pdf["d"].astype(np.float64)

    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float32)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    # 10. any float -> any int raises
    pdf["a"] = pdf["a"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["a"] = pdf["a"].astype(np.int32)
    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["b"] = pdf["b"].astype(np.int64)

    pdf["b"] = pdf["b"].astype(np.float64)
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(pdf)
    pdf["b"] = pdf["b"].astype(np.int64)

    # 11. objects work
    pdf["b"] = pdf["b"].astype(np.object)
    pdf["d"] = pdf["d"].astype(np.object)
    pdf["e"] = pdf["e"].astype(np.object)
    pdf["f"] = pdf["f"].astype(np.object)
    pdf["g"] = pdf["g"].astype(np.object)
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types

    # 12. datetime64[D] (date only) -> datetime64[x] works
    pdf["h"] = pdf["h"].astype("datetime64[D]")
    res = pyfunc_model.predict(pdf)
    assert res.dtypes.to_dict() == expected_types
    pdf["h"] = pdf["h"].astype("datetime64[s]")

    # 13. np.ndarrays can be converted to dataframe but have no columns
    with pytest.raises(MlflowException, match=match_missing_inputs):
        pyfunc_model.predict(pdf.values)

    # 14. dictionaries of str -> list/nparray work
    arr = np.array([1, 2, 3])
    d = {
        "a": arr.astype("int32"),
        "b": arr.astype("int64"),
        "c": arr.astype("float32"),
        "d": arr.astype("float64"),
        "e": [True, False, True],
        "g": ["a", "b", "c"],
        "f": [bytes(0), bytes(1), bytes(1)],
        "h": np.array(["2020-01-01", "2020-02-02", "2020-03-03"], dtype=np.datetime64),
    }
    res = pyfunc_model.predict(d)
    assert res.dtypes.to_dict() == expected_types

    # 15. dictionaries of str -> list[list] fail
    d = {
        "a": [arr.astype("int32")],
        "b": [arr.astype("int64")],
        "c": [arr.astype("float32")],
        "d": [arr.astype("float64")],
        "e": [[True, False, True]],
        "g": [["a", "b", "c"]],
        "f": [[bytes(0), bytes(1), bytes(1)]],
        "h": [np.array(["2020-01-01", "2020-02-02", "2020-03-03"], dtype=np.datetime64)],
    }
    with pytest.raises(MlflowException, match=match_incompatible_inputs):
        pyfunc_model.predict(d)

    # 16. conversion to dataframe fails
    d = {
        "a": [1],
        "b": [1, 2],
        "c": [1, 2, 3],
    }
    with pytest.raises(
        MlflowException,
        match="This model contains a column-based signature, which suggests a DataFrame input.",
    ):
        pyfunc_model.predict(d)


def _compare_exact_tensor_dict_input(d1, d2):
    """Return whether two dicts of np arrays are exactly equal"""
    if d1.keys() != d2.keys():
        return False
    return all(np.array_equal(d1[key], d2[key]) for key in d1)


def test_tensor_multi_named_schema_enforcement():
    m = Model()
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.uint64), (-1, 5), "a"),
            TensorSpec(np.dtype(np.short), (-1, 2), "b"),
            TensorSpec(np.dtype(np.float32), (2, -1, 2), "c"),
        ]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    inp = {
        "a": np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.uint64),
        "b": np.array([[0, 0], [1, 1], [2, 2]], dtype=np.short),
        "c": np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]], dtype=np.float32),
    }

    # test that missing column raises
    inp1 = {k: v for k, v in inp.items()}
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        pyfunc_model.predict(inp1.pop("b"))

    # test that extra column is ignored
    inp2 = {k: v for k, v in inp.items()}
    inp2["x"] = 1

    # test that extra column is removed
    res = pyfunc_model.predict(inp2)
    assert res == {k: v for k, v in inp.items() if k in {"a", "b", "c"}}
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test that variable axes are supported
    inp3 = {
        "a": np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]], dtype=np.uint64),
        "b": np.array([[0, 0], [1, 1]], dtype=np.short),
        "c": np.array([[[0, 0]], [[2, 2]]], dtype=np.float32),
    }
    res = pyfunc_model.predict(inp3)
    assert _compare_exact_tensor_dict_input(res, inp3)
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test that type casting is not supported
    inp4 = {k: v for k, v in inp.items()}
    inp4["a"] = inp4["a"].astype(np.int32)
    with pytest.raises(
        MlflowException, match="dtype of input int32 does not match expected dtype uint64"
    ):
        pyfunc_model.predict(inp4)

    # test wrong shape
    inp5 = {
        "a": np.array([[0, 0, 0, 0]], dtype=np.uint),
        "b": np.array([[0, 0], [1, 1]], dtype=np.short),
        "c": np.array([[[0, 0]]], dtype=np.float32),
    }
    with pytest.raises(
        MlflowException,
        match=re.escape("Shape of input (1, 4) does not match expected shape (-1, 5)"),
    ):
        pyfunc_model.predict(inp5)

    # test non-dictionary input
    inp6 = [
        np.array([[0, 0, 0, 0, 0]], dtype=np.uint64),
        np.array([[0, 0], [1, 1]], dtype=np.short),
        np.array([[[0, 0]]], dtype=np.float32),
    ]
    with pytest.raises(
        MlflowException, match=re.escape("Model is missing inputs ['a', 'b', 'c'].")
    ):
        pyfunc_model.predict(inp6)

    # test empty ndarray does not work
    inp7 = {k: v for k, v in inp.items()}
    inp7["a"] = np.array([])
    with pytest.raises(
        MlflowException, match=re.escape("Shape of input (0,) does not match expected shape")
    ):
        pyfunc_model.predict(inp7)

    # test dictionary of str -> list does not work
    inp8 = {k: list(v) for k, v in inp.items()}
    match = (
        r"This model contains a tensor-based model signature with input names.+"
        r"suggests a dictionary input mapping input name to a numpy array, but a dict"
        r" with value type <class 'list'> was found"
    )
    with pytest.raises(MlflowException, match=match):
        pyfunc_model.predict(inp8)

    # test dataframe input fails at shape enforcement
    pdf = pd.DataFrame(data=[[1, 2, 3]], columns=["a", "b", "c"])
    pdf["a"] = pdf["a"].astype(np.uint64)
    pdf["b"] = pdf["b"].astype(np.short)
    pdf["c"] = pdf["c"].astype(np.float32)
    with pytest.raises(
        MlflowException,
        match=re.escape("Shape of input (1,) does not match expected shape (-1, 5)"),
    ):
        pyfunc_model.predict(pdf)


def test_schema_enforcement_single_named_tensor_schema():
    m = Model()
    input_schema = Schema([TensorSpec(np.dtype(np.uint64), (-1, 2), "a")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    inp = {
        "a": np.array([[0, 0], [1, 1]], dtype=np.uint64),
    }

    # sanity test that dictionary with correct input works
    res = pyfunc_model.predict(inp)
    assert res == inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test single np.ndarray input works and is converted to dictionary
    res = pyfunc_model.predict(inp["a"])
    assert res == inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test list does not work
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        pyfunc_model.predict([[0, 0], [1, 1]])


def test_schema_enforcement_named_tensor_schema_1d():
    m = Model()
    input_schema = Schema(
        [TensorSpec(np.dtype(np.uint64), (-1,), "a"), TensorSpec(np.dtype(np.float32), (-1,), "b")]
    )
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[0, 0], [1, 1]], columns=["a", "b"])
    pdf["a"] = pdf["a"].astype(np.uint64)
    pdf["b"] = pdf["a"].astype(np.float32)
    d_inp = {
        "a": np.array(pdf["a"], dtype=np.uint64),
        "b": np.array(pdf["b"], dtype=np.float32),
    }

    # test dataframe input works for 1d tensor specs and input is converted to dict
    res = pyfunc_model.predict(pdf)
    assert _compare_exact_tensor_dict_input(res, d_inp)
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types

    # test that dictionary works too
    res = pyfunc_model.predict(d_inp)
    assert res == d_inp
    expected_types = dict(zip(input_schema.input_names(), input_schema.input_types()))
    actual_types = {k: v.dtype for k, v in res.items()}
    assert expected_types == actual_types


def test_missing_value_hint_is_displayed_when_it_should():
    m = Model()
    input_schema = Schema([ColSpec("integer", "a")])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    pdf = pd.DataFrame(data=[[1], [None]], columns=["a"])
    match = "Incompatible input types"
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    hint = "Hint: the type mismatch is likely caused by missing values."
    assert hint in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1.5], [None]], columns=["a"])
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    assert hint not in str(ex.value.message)
    pdf = pd.DataFrame(data=[[1], [2]], columns=["a"], dtype=np.float64)
    with pytest.raises(MlflowException, match=match) as ex:
        pyfunc_model.predict(pdf)
    assert hint not in str(ex.value.message)


def test_column_schema_enforcement_no_col_names():
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
    with pytest.raises(MlflowException, match="the provided value only has 2 inputs."):
        pyfunc_model.predict([[1.0, 2.0]])

    # Must provide the right types
    with pytest.raises(MlflowException, match="Can not safely convert int64 to float64"):
        pyfunc_model.predict([[1, 2, 3]])

    # Can only provide data type that can be converted to dataframe...
    with pytest.raises(MlflowException, match="Expected input to be DataFrame or list. Found: set"):
        pyfunc_model.predict(set([1, 2, 3]))

    # 9. dictionaries of str -> list/nparray work
    d = {"a": [1.0], "b": [2.0], "c": [3.0]}
    assert pyfunc_model.predict(d).equals(pd.DataFrame(d))


def test_tensor_schema_enforcement_no_col_names():
    m = Model()
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3))])
    m.signature = ModelSignature(inputs=input_schema)
    pyfunc_model = PyFuncModel(model_meta=m, model_impl=TestModel())
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Can call with numpy array of correct shape
    assert np.array_equal(pyfunc_model.predict(test_data), test_data)

    # Or can call with a dataframe
    assert np.array_equal(pyfunc_model.predict(pd.DataFrame(test_data)), test_data)

    # Can not call with a list
    with pytest.raises(
        MlflowException,
        match="This model contains a tensor-based model signature with no input names",
    ):
        pyfunc_model.predict([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Can not call with a dict
    with pytest.raises(
        MlflowException,
        match="This model contains a tensor-based model signature with no input names",
    ):
        pyfunc_model.predict({"blah": test_data})

    # Can not call with a np.ndarray of a wrong shape
    with pytest.raises(
        MlflowException,
        match=re.escape("Shape of input (2, 2) does not match expected shape (-1, 3)"),
    ):
        pyfunc_model.predict(np.array([[1.0, 2.0], [4.0, 5.0]]))

    # Can not call with a np.ndarray of a wrong type
    with pytest.raises(
        MlflowException, match="dtype of input uint32 does not match expected dtype float32"
    ):
        pyfunc_model.predict(test_data.astype(np.uint32))

    # Can call with a np.ndarray with more elements along variable axis
    test_data2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    assert np.array_equal(pyfunc_model.predict(test_data2), test_data2)

    # Can not call with an empty ndarray
    with pytest.raises(
        MlflowException, match=re.escape("Shape of input () does not match expected shape (-1, 3)")
    ):
        pyfunc_model.predict(np.ndarray([]))


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
            loader_module=__name__,
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
        loader_module=__name__,
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
    with pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.save_model(path=model_path, data_path="/path/to/data")


@pytest.mark.large
def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model", data_path="/path/to/data")


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
            loader_module=__name__,
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
            loader_module=__name__,
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
def test_log_model_persists_requirements_in_mlflow_model_directory(
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
            loader_module=__name__,
            code_path=[__file__],
            conda_env=pyfunc_custom_env_dict,
        )
        run_id = mlflow.active_run().info.run_id

    pyfunc_model_path = _download_artifact_from_uri(
        "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=pyfunc_artifact_path)
    )

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    assert os.path.exists(saved_pip_req_path)

    with open(saved_pip_req_path, "r") as f:
        requirements = f.read().split("\n")

    assert pyfunc_custom_env_dict["dependencies"][-1]["pip"] == requirements


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
            loader_module=__name__,
            code_path=[__file__],
        )
        model_uri = mlflow.get_artifact_uri(pyfunc_artifact_path)
    _assert_pip_requirements(model_uri, mlflow.pyfunc.get_default_pip_requirements())
