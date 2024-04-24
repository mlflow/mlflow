import pickle
import mlflow
import os
from mlflow.models import Model
import numpy as np
import pytest
import sklearn.neighbors
import sklearn.datasets
import sys

sys.path.insert(0, ".")


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


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
    for root, dirs, files in os.walk(path):
        for file in files:
            yield os.path.relpath(os.path.join(root, file), start=path)


def test_loader_module_model_save_load(sklearn_knn_model, iris_data, tmp_path, model_path):
    sk_model_path = os.path.join(tmp_path, "knn.pkl")
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

    reloaded_model_config = Model.load(os.path.join(model_path, "MLmodel"))

    assert set(_walk_dir(os.path.join(model_path, "code"))) == {
        'custom_model/loader.py',
        'custom_model/mod1/__init__.py',
        'custom_model/mod1/mod2/__init__.py',
        'custom_model/mod1/mod4.py'
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
        def __init__(self, predict_fn):
            self.predict_fn = predict_fn
            self.mod2 = mod2

        def load_context(self, context):
            super().load_context(context)

            self.model = mlflow.sklearn.load_model(model_uri=context.artifacts["sk_model"])

        def predict(self, context, model_input, params=None):
            return self.predict_fn(self.model, model_input)

    return CustomSklearnModel


def test_python_model_save_load(sklearn_knn_model, iris_data, tmp_path):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    model_class = get_model_class()

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=model_class(test_predict),
        infer_code_paths=True,
    )

    assert set(_walk_dir(os.path.join(pyfunc_model_path, "code"))) == {
        'custom_model/mod1/__init__.py',
        'custom_model/mod1/mod2/__init__.py',
        'custom_model/mod1/mod4.py'
    }
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )
