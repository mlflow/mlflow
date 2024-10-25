"""Functions for loading Keras models saved with MLflow."""

import os

import keras
import numpy as np
import pandas as pd

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental

_MODEL_SAVE_PATH = "model"


class KerasModelWrapper:
    def __init__(self, model, signature, save_exported_model=False):
        self.model = model
        self.signature = signature
        self.save_exported_model = save_exported_model

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.model

    def get_model_call_method(self):
        if self.save_exported_model:
            return self.model.serve
        else:
            return self.model.predict

    def predict(self, data, **kwargs):
        model_call = self.get_model_call_method()
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(model_call(data.values), index=data.index)

        supported_input_types = (np.ndarray, list, tuple, dict)
        if not isinstance(data, supported_input_types):
            raise MlflowException(
                f"`data` must be one of: {[x.__name__ for x in supported_input_types]}, but "
                f"received type: {type(data)}.",
                INVALID_PARAMETER_VALUE,
            )
        # Return numpy array for serving purposes.
        return keras.ops.convert_to_numpy(model_call(data))


def _load_keras_model(path, model_conf, custom_objects=None, **load_model_kwargs):
    save_exported_model = model_conf.flavors["keras"].get("save_exported_model")
    model_path = os.path.join(path, model_conf.flavors["keras"].get("data", _MODEL_SAVE_PATH))
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if save_exported_model:
        try:
            import tensorflow as tf
        except ImportError:
            raise MlflowException(
                "`tensorflow` must be installed if you want to load an exported Keras 3 model, "
                "please install `tensorflow` by `pip install tensorflow`."
            )
        return tf.saved_model.load(model_path)
    else:
        model_path += ".keras"
        return keras.saving.load_model(
            model_path,
            custom_objects=custom_objects,
            **load_model_kwargs,
        )


@experimental
def load_model(model_uri, dst_path=None, custom_objects=None, load_model_kwargs=None):
    """
    Load Keras model from MLflow.

    This method loads a saved Keras model from MLflow, and returns a Keras model instance.

    Args:
        model_uri: The URI of the saved Keras model in MLflow. For example:

            - `/Users/me/path/to/local/model`
            - `relative/path/to/local/model`
            - `s3://my_bucket/path/to/model`
            - `runs:/<mlflow_run_id>/run-relative/path/to/model`
            - `models:/<model_name>/<model_version>`
            - `models:/<model_name>/<stage>`

            For more information about supported URI schemes, see `Referencing
            Artifacts <https://www.mlflow.org/docs/latest/concepts.html#artifact-locations>`_.
        dst_path: The local filesystem path to which to download the
            model artifact. If unspecified, a local output path will be created.
        custom_objects: The `custom_objects` arg in
            `keras.saving.load_model`.
        load_model_kwargs: Extra args for `keras.saving.load_model`.

    .. code-block:: python
        :caption: Example

        import keras
        import mlflow
        import numpy as np

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )
        with mlflow.start_run() as run:
            mlflow.keras.log_model(model)

        model_url = f"runs:/{run.info.run_id}/{model_path}"
        loaded_model = mlflow.keras.load_model(model_url)

        # Test the loaded model produces the same output for the same input as the model.
        test_input = np.random.uniform(size=[2, 28, 28, 3])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model(test_input)),
            loaded_model.predict(test_input),
        )

    Returns:
        A Keras model instance.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    load_model_kwargs = {} if load_model_kwargs is None else load_model_kwargs

    model_configuration_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)
    model_conf = Model.load(model_configuration_path)

    return _load_keras_model(local_model_path, model_conf, custom_objects, **load_model_kwargs)


def _load_pyfunc(path):
    """Logics of loading a saved Keras model as a PyFunc model.

    This function is called by `mlflow.pyfunc.load_model`.

    Args:
        path: Local filesystem path to the MLflow Model with the `keras` flavor.
    """
    model_meta_path1 = os.path.join(path, MLMODEL_FILE_NAME)
    model_meta_path2 = os.path.join(os.path.dirname(path), MLMODEL_FILE_NAME)

    if os.path.isfile(model_meta_path1):
        model_conf = Model.load(model_meta_path1)
    elif os.path.isfile(model_meta_path2):
        model_conf = Model.load(model_meta_path2)
    else:
        raise MlflowException(f"Cannot find file {MLMODEL_FILE_NAME} for the logged model.")

    save_exported_model = model_conf.flavors["keras"].get("save_exported_model")

    loaded_model = _load_keras_model(path, model_conf)
    return KerasModelWrapper(loaded_model, model_conf.signature, save_exported_model)
