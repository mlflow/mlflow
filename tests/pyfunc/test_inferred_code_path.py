import os
import pickle

import numpy as np
import pytest
import sklearn.datasets
import sklearn.neighbors

import mlflow
from mlflow.models import Model


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


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


def _walk_dir(path):
    return {
        str(p.relative_to(path))
        for p in path.rglob("*")
        if p.is_file() and p.parent.name != "__pycache__"
    }


def test_loader_module_model_save_load(
    sklearn_knn_model, iris_data, tmp_path, model_path, monkeypatch
):
    monkeypatch.chdir(os.path.dirname(__file__))
    monkeypatch.syspath_prepend(".")
    sk_model_path = tmp_path / "knn.pkl"
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=sk_model_path,
        loader_module="custom_model.loader",
        mlflow_model=model_config,
        infer_code_paths=True,
    )

    reloaded_model_config = Model.load(model_path / "MLmodel")

    assert _walk_dir(model_path / "code") == {
        "custom_model/loader.py",
        "custom_model/mod1/__init__.py",
        "custom_model/mod1/mod2/__init__.py",
        "custom_model/mod1/mod4.py",
    }
    assert model_config.__dict__ == reloaded_model_config.__dict__
    assert mlflow.pyfunc.FLAVOR_NAME in reloaded_model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in reloaded_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0])
    )


def get_model_class():
    """
    Defines a custom Python model class that wraps a scikit-learn estimator.
    This can be invoked within a pytest fixture to define the class in the ``__main__`` scope.
    Alternatively, it can be invoked within a module to define the class in the module's scope.
    """
    from custom_model.mod1 import mod2

    class CustomSklearnModel(mlflow.pyfunc.PythonModel):
        def __init__(self):
            self.mod2 = mod2

        def predict(self, context, model_input, params=None):
            return [x + 10 for x in model_input]

    return CustomSklearnModel


def test_python_model_save_load(tmp_path, monkeypatch):
    monkeypatch.chdir(os.path.dirname(__file__))
    monkeypatch.syspath_prepend(".")

    model_class = get_model_class()

    pyfunc_model_path = tmp_path / "pyfunc_model"

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        python_model=model_class(),
        infer_code_paths=True,
    )

    assert _walk_dir(pyfunc_model_path / "code") == {
        "custom_model/mod1/__init__.py",
        "custom_model/mod1/mod2/__init__.py",
        "custom_model/mod1/mod4.py",
    }
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict([1, 2, 3]),
        [11, 12, 13],
    )
