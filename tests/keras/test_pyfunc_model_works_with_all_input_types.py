# pep8: disable=E501


from distutils.version import LooseVersion
import os
import pytest

import keras
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
    # Use a small learning rate to prevent exploding gradients which may produce
    # infinite prediction values
    lr = 0.001
    kwargs = (
        # `lr` was renamed to `learning_rate` in keras 2.3.0:
        # https://github.com/keras-team/keras/releases/tag/2.3.0
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit(x.values, y.values)
    return model


@pytest.fixture(scope="module")
def multi_tensor_input_model_list(data):
    x, y = data
    input_a = Input(shape=(4,))
    input_b = Input(shape=(4,))
    output = Dense(1)(Concatenate()([input_a, input_b]))
    model = Model(inputs=[input_a, input_b], outputs=output)
    lr = 0.001
    kwargs = (
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit([x.values, x.values], y)
    return model


@pytest.fixture(scope="module")
def multi_tensor_input_model_dict(data):
    x, y = data
    input_a = Input(shape=(4,))
    input_b = Input(shape=(4,))
    output = Dense(1)(Concatenate()([input_a, input_b]))
    model = Model(inputs={"a": input_a, "b": input_b}, outputs=output)
    lr = 0.001
    kwargs = (
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit({"a": x.values, "b": x.values}, y)
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

    # Calling predict with a dict should make keras raise an error
    with pytest.raises(Exception):
        model_loaded.predict({"a": [1, 2, 3], "b": [2, 3, 4]})


def test_model_multi_tensor_input_list(multi_tensor_input_model_list, model_path, data):
    x, _ = data
    test_input = [x.values, x.values]

    model_path = os.path.join(model_path, "plain")
    expected = multi_tensor_input_model_list.predict(test_input)
    mlflow.keras.save_model(multi_tensor_input_model_list, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a list should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a non-list input should raise an error
    with pytest.raises(Exception):
        model_loaded.predict({"a": [1], "b": [2]})


def test_model_multi_tensor_input_dict(multi_tensor_input_model_dict, model_path, data):
    x, _ = data
    test_input = {
        "a": x.values,
        "b": x.values,
    }

    model_path = os.path.join(model_path, "plain")
    expected = multi_tensor_input_model_dict.predict(test_input)
    mlflow.keras.save_model(multi_tensor_input_model_dict, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a dict should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a non-dict input should raise an error
    with pytest.raises(Exception):
        model_loaded.predict([1, 2, 3, 4])
