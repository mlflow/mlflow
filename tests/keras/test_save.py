import keras
import numpy as np
import pytest

import mlflow
from mlflow.keras.utils import get_model_signature
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def _get_keras_model():
    return keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )


def test_keras_save_model_export():
    if keras.backend.backend() == "torch":
        pytest.skip("Keras model exporting is not supported in torch backend.")

    model = _get_keras_model()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(0.002),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model_path = "model"
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 28, 28, 3))])
    signature = ModelSignature(inputs=input_schema)
    with mlflow.start_run() as run:
        mlflow.keras.log_model(
            model,
            model_path,
            save_exported_model=True,
            signature=signature,
        )

    model_url = f"runs:/{run.info.run_id}/{model_path}"
    loaded_model = mlflow.keras.load_model(model_url)

    # Test the loaded model produces the same output for the same input as the model.
    test_input = np.random.uniform(size=[2, 28, 28, 3]).astype(np.float32)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(test_input)),
        loaded_model.serve(test_input),
    )

    # Test the loaded pyfunc model produces the same output for the same input as the model.
    logged_model = f"runs:/{run.info.run_id}/{model_path}"
    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(test_input)),
        loaded_pyfunc_model.predict(test_input),
    )


def test_keras_save_model_non_export():
    model = _get_keras_model()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(0.002),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model_path = "model"
    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, "model", save_exported_model=False)

    model_url = f"runs:/{run.info.run_id}/{model_path}"
    loaded_model = mlflow.keras.load_model(model_url)

    # Test the loaded model produces the same output for the same input as the model.
    test_input = np.random.uniform(size=[2, 28, 28, 3])
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(test_input)),
        loaded_model.predict(test_input),
    )

    assert loaded_model.optimizer.name == "adam"
    assert loaded_model.optimizer.learning_rate == model.optimizer.learning_rate

    # Test the loaded pyfunc model produces the same output for the same input as the model.
    logged_model = f"runs:/{run.info.run_id}/{model_path}"
    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(test_input)),
        loaded_pyfunc_model.predict(test_input),
    )


def test_save_model_with_signature():
    keras.mixed_precision.set_dtype_policy("mixed_float16")

    model = _get_keras_model()
    signature = get_model_signature(model)
    assert signature.outputs.input_types()[0] == np.dtype("float16")
    model_path = "model"
    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, model_path, signature=signature)

    logged_model = f"runs:/{run.info.run_id}/{model_path}"
    loaded_pyfunc_model = mlflow.pyfunc.load_model(logged_model)

    assert signature == loaded_pyfunc_model.metadata.signature

    # Test the loaded model produces the same output for the same input as the model.
    test_input = np.random.uniform(size=[2, 28, 28, 3]).astype(np.float32)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(test_input)),
        loaded_pyfunc_model.predict(test_input),
    )

    # Clean up the global policy.
    keras.mixed_precision.set_dtype_policy("float32")
