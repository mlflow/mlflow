import os
from typing import Any, NamedTuple
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import mlflow.tensorflow
from mlflow.models import Model, infer_signature


class ToyModel(tf.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = w
        self.b = b

    @tf.function
    def __call__(self, x):
        return tf.reshape(tf.add(tf.matmul(x, self.w), self.b), [-1])


class TF2ModelInfo(NamedTuple):
    model: Any
    inference_data: Any
    expected_results: Any


@pytest.fixture
def tf2_toy_model():
    tf.random.set_seed(1337)
    rand_w = tf.random.uniform(shape=[3, 1], dtype=tf.float32)
    rand_b = tf.random.uniform(shape=[], dtype=tf.float32)

    inference_data = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    model = ToyModel(rand_w, rand_b)
    expected_results = model(inference_data)

    return TF2ModelInfo(
        model=model,
        inference_data=inference_data,
        expected_results=expected_results,
    )


def test_save_and_load_tf2_module(tmp_path, tf2_toy_model):
    model_path = os.path.join(tmp_path, "model")
    mlflow.tensorflow.save_model(tf2_toy_model.model, model_path)

    loaded_model = mlflow.tensorflow.load_model(model_path)

    predictions = loaded_model(tf2_toy_model.inference_data).numpy()
    np.testing.assert_allclose(
        predictions,
        tf2_toy_model.expected_results,
    )


def test_log_and_load_tf2_module(tf2_toy_model):
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(tf2_toy_model.model, name="model")

    model_uri = model_info.model_uri
    loaded_model = mlflow.tensorflow.load_model(model_uri)
    predictions = loaded_model(tf2_toy_model.inference_data).numpy()
    np.testing.assert_allclose(
        predictions,
        tf2_toy_model.expected_results,
    )

    loaded_model2 = mlflow.pyfunc.load_model(model_uri)
    predictions2 = loaded_model2.predict(tf2_toy_model.inference_data)
    assert isinstance(predictions2, np.ndarray)
    np.testing.assert_allclose(
        predictions2,
        tf2_toy_model.expected_results,
    )


def test_model_log_with_signature_inference(tf2_toy_model):
    artifact_path = "model"
    example = tf2_toy_model.inference_data

    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            tf2_toy_model.model, name=artifact_path, input_example=example
        )

    mlflow_model = Model.load(model_info.model_uri)
    assert mlflow_model.signature == infer_signature(
        tf2_toy_model.inference_data, tf2_toy_model.expected_results.numpy()
    )


def test_save_with_options(tmp_path, tf2_toy_model):
    model_path = os.path.join(tmp_path, "model")

    saved_model_kwargs = {
        "signatures": [tf.TensorSpec(shape=None, dtype=tf.float32)],
        "options": tf.saved_model.SaveOptions(save_debug_info=True),
    }

    with mock.patch("tensorflow.saved_model.save") as mock_save:
        mlflow.tensorflow.save_model(
            tf2_toy_model.model, model_path, saved_model_kwargs=saved_model_kwargs
        )
        mock_save.assert_called_once_with(mock.ANY, mock.ANY, **saved_model_kwargs)

        mock_save.reset_mock()

        with mlflow.start_run():
            mlflow.tensorflow.log_model(
                tf2_toy_model.model, name="model", saved_model_kwargs=saved_model_kwargs
            )

        mock_save.assert_called_once_with(mock.ANY, mock.ANY, **saved_model_kwargs)


def test_load_with_options(tmp_path, tf2_toy_model):
    model_path = os.path.join(tmp_path, "model")
    mlflow.tensorflow.save_model(tf2_toy_model.model, model_path)

    saved_model_kwargs = {
        "options": tf.saved_model.LoadOptions(),
    }
    with mock.patch("tensorflow.saved_model.load") as mock_load:
        mlflow.tensorflow.load_model(model_path, saved_model_kwargs=saved_model_kwargs)
        mock_load.assert_called_once_with(mock.ANY, **saved_model_kwargs)
