# pep8: disable=E501


import os
import pytest

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import SGD
import sklearn.datasets as datasets
import pandas as pd
import numpy as np

import mlflow


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture(scope="module")
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    x = data.drop("target", axis=1)
    return x, y


@pytest.fixture(scope="module")
def single_tensor_input_model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit(x.values, y.values)
    return model


@pytest.fixture(scope="module")
def multi_tensor_input_model(data):
    x, y = data
    input_a = Input(shape=(2,), name="a")
    input_b = Input(shape=(2,), name="b")
    output = Dense(1)(Dense(3, input_dim=4)(Concatenate()([input_a, input_b])))
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit([x.values[:, :2], x.values[:, -2:]], y)
    return model


def test_model_single_tensor_input(single_tensor_input_model, model_path, data):
    x, _ = data
    model_path = os.path.join(model_path, "plain")
    expected = single_tensor_input_model.predict(x.values)
    mlflow.keras.save_model(single_tensor_input_model, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a dataframe should return a dataframe
    actual = model_loaded.predict(x)
    assert type(actual) == pd.DataFrame
    np.testing.assert_allclose(actual.values, expected, rtol=1e-5)

    # Calling predict with a np array should return a np array
    actual = model_loaded.predict(x.values)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_model_multi_tensor_input(multi_tensor_input_model, model_path, data):
    x, _ = data
    test_input = [x.values[:, :2], x.values[:, -2:]]

    model_path = os.path.join(model_path, "plain")
    expected = multi_tensor_input_model.predict(test_input)
    mlflow.keras.save_model(multi_tensor_input_model, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a list should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a dict should return a np.ndarray output
    test_input = {
        "a": x.values[:, :2],
        "b": x.values[:, -2:],
    }
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
