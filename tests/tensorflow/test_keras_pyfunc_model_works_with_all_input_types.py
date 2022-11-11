import os
import pytest

from sklearn import datasets
import pandas as pd
import numpy as np
import json

import mlflow

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import SGD


from mlflow.types.schema import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from tests.helper_functions import pyfunc_serve_and_score_model, expect_status_code


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

    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float64), (-1, 4)),
            ]
        )
    )
    return model, signature


@pytest.fixture(scope="module")
def multi_tensor_input_model(data):
    x, y = data
    input_a = Input(shape=(2,), name="a")
    input_b = Input(shape=(2,), name="b")
    output = Dense(1)(Dense(3, input_dim=4)(Concatenate()([input_a, input_b])))
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit([x.values[:, :2], x.values[:, -2:]], y)

    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float64), (-1, 2), "a"),
                TensorSpec(np.dtype(np.float64), (-1, 2), "b"),
            ]
        )
    )
    return model, signature


@pytest.fixture(scope="module")
def single_multidim_tensor_input_model(data):
    """
    This is a model that requires a single input of shape (-1, 4, 3)
    """
    x, y = data
    model = Sequential()

    def f1(z):
        from tensorflow.keras import backend as K

        return K.mean(z, axis=2)

    model.add(Lambda(f1))
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit(np.repeat(x.values[:, :, np.newaxis], 3, axis=2), y.values)
    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float64), (-1, 4, 3)),
            ]
        )
    )
    return model, signature


@pytest.fixture(scope="module")
def multi_multidim_tensor_input_model(data):
    """
    This is a model that requires 2 inputs: 'a' and 'b',
    input 'a' must be shape of (-1, 2, 3),
    input 'b' must be shape of (-1, 2, 5),
    """
    x, y = data
    input_a = Input(shape=(2, 3), name="a")
    input_b = Input(shape=(2, 5), name="b")

    def f1(z):
        from tensorflow.keras import backend as K

        return K.mean(z, axis=2)

    input_a_sum = Lambda(f1)(input_a)
    input_b_sum = Lambda(f1)(input_b)

    output = Dense(1)(Dense(3, input_dim=4)(Concatenate()([input_a_sum, input_b_sum])))
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit(
        [
            np.repeat(x.values[:, :2, np.newaxis], 3, axis=2),
            np.repeat(x.values[:, -2:, np.newaxis], 5, axis=2),
        ],
        y,
    )
    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float64), (-1, 2, 3), "a"),
                TensorSpec(np.dtype(np.float64), (-1, 2, 5), "b"),
            ]
        )
    )
    return model, signature


def test_model_single_tensor_input(single_tensor_input_model, model_path, data):
    x, _ = data
    model, signature = single_tensor_input_model
    model_path = os.path.join(model_path, "plain")
    expected = model.predict(x.values)
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    actual = model_loaded.predict(x)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a np array should return a np array
    actual = model_loaded.predict(x.values)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_model_multi_tensor_input(multi_tensor_input_model, model_path, data):
    x, _ = data

    model, signature = multi_tensor_input_model
    test_input = {
        "a": x.values[:, :2],
        "b": x.values[:, -2:],
    }

    model_path = os.path.join(model_path, "plain")
    expected = model.predict(test_input)
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a list should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    test_input = pd.DataFrame(
        {
            "a": list(x.values[:, :2]),
            "b": list(x.values[:, -2:]),
        }
    )
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_model_single_multidim_tensor_input(single_multidim_tensor_input_model, model_path, data):
    x, _ = data
    model, signature = single_multidim_tensor_input_model
    test_input = np.repeat(x.values[:, :, np.newaxis], 3, axis=2)
    model_path = os.path.join(model_path, "plain")
    expected = model.predict(test_input)
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    test_input_df = pd.DataFrame({"x": list(test_input.reshape((-1, 4 * 3)))})
    actual = model_loaded.predict(test_input_df)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_model_multi_multidim_tensor_input(multi_multidim_tensor_input_model, model_path, data):
    x, _ = data
    model, signature = multi_multidim_tensor_input_model
    input_a = np.repeat(x.values[:, :2, np.newaxis], 3, axis=2)
    input_b = np.repeat(x.values[:, -2:, np.newaxis], 5, axis=2)
    test_input = {
        "a": input_a,
        "b": input_b,
    }

    model_path = os.path.join(model_path, "plain")
    expected = model.predict(test_input)
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    test_input = pd.DataFrame(
        {
            "a": list(input_a.reshape((-1, 2 * 3))),
            "b": list(input_b.reshape((-1, 2 * 5))),
        }
    )
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_scoring_server_successfully_evaluates_correct_tf_serving_single_multidim_input_model(
    single_multidim_tensor_input_model, model_path, data
):
    model, signature = single_multidim_tensor_input_model
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    x, _ = data

    test_input = np.repeat(x.values[:, :, np.newaxis], 3, axis=2)

    inp_dict = {"instances": test_input.tolist()}
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)


def test_scoring_server_successfully_evaluates_correct_tf_serving_multi_multidim_input_model(
    multi_multidim_tensor_input_model, model_path, data
):
    model, signature = multi_multidim_tensor_input_model
    mlflow.tensorflow.save_model(model, path=model_path, signature=signature)

    x, _ = data

    input_a = np.repeat(x.values[:, :2, np.newaxis], 3, axis=2)
    input_b = np.repeat(x.values[:, -2:, np.newaxis], 5, axis=2)

    instances = []
    for index in range(len(input_a)):
        instances.append(
            {
                "a": input_a[index].tolist(),
                "b": input_b[index].tolist(),
            }
        )

    inp_dict = {"instances": instances}
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    expect_status_code(response_records_content_type, 200)
