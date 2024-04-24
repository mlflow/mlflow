"""MLflow autologging support for Keras 3."""

import logging

import keras
import numpy as np

import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.exceptions import MlflowException
from mlflow.keras.callback import MlflowCallback
from mlflow.keras.save import log_model
from mlflow.keras.utils import get_model_signature
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import is_iterator
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    PatchFunction,
    autologging_integration,
    get_autologging_config,
    log_fn_args_as_params,
    safe_patch,
)

_logger = logging.getLogger(__name__)


def _check_existing_mlflow_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, MlflowCallback):
            raise MlflowException(
                "MLflow autologging must be turned off if an `MlflowCallback` is explicitly added "
                "to the callback list. You are creating an `MlflowCallback` while having "
                "autologging enabled. Please either call `mlflow.keras.autolog(disable=True)` "
                "to disable autologging or remove `MlflowCallback` from the callback list. "
            )


def _log_dataset(dataset, source, context, name=None, targets=None):
    """Helper function to log the dataset information to MLflow."""
    try:
        import tensorflow as tf

        is_tf_dataset = isinstance(dataset, tf.data.Dataset)
        is_tf_tensor = isinstance(dataset, tf.Tensor)
    except ImportError:
        pass

    if isinstance(dataset, np.ndarray):
        dataset = from_numpy(features=dataset, targets=targets, source=source, name=name)
    elif is_tf_tensor:
        dataset = from_tensorflow(features=dataset, targets=targets, source=source, name=name)
    elif is_tf_dataset:
        dataset = from_tensorflow(features=dataset, source=source, name=name)
    elif isinstance(dataset, tuple):
        x = dataset[0]
        y = dataset[1]
        # check if x and y are tensors
        if isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
            dataset = from_tensorflow(features=x, source=source, targets=y, name=name)
        else:
            dataset = from_numpy(features=x, targets=y, source=source, name=name)
    else:
        _logger.warning(f"Unrecognized dataset type {type(dataset)}. Dataset logging skipped.")
        return

    mlflow.log_input(dataset, context)


def _parse_dataset(*keras_fit_args, **keras_fit_kwargs):
    """Parse dataset from `keras.Model.fit` args and kwargs."""
    x = keras_fit_kwargs["x"] if "x" in keras_fit_kwargs else keras_fit_args[0]
    if "y" in keras_fit_kwargs:
        # `y` is either specified as a kwarg, or the second argument to `fit`.
        y = keras_fit_kwargs["y"]
    elif len(keras_fit_args) >= 2:
        y = keras_fit_args[1]
    else:
        y = None
    return x, y


def _log_keras_model(
    model,
    save_exported_model,
    log_model_signatures=True,
    save_model_kwargs=None,
):
    """Helper function to log the Keras model to MLflow."""
    if log_model_signatures:
        try:
            signature = get_model_signature(model)
        except Exception as e:
            _logger.warning(f"Failed to get model signature, reason: {e}")
            signature = None
    else:
        signature = None

    log_model(
        model=model,
        artifact_path="model",
        save_exported_model=save_exported_model,
        signature=signature,
        registered_model_name=get_autologging_config("keras", "registered_model_name", None),
        save_model_kwargs=save_model_kwargs,
    )


@experimental
@autologging_integration("keras")
def autolog(
    log_every_epoch=True,
    log_every_n_steps=None,
    log_models=True,
    log_model_signatures=True,
    save_exported_model=False,
    log_datasets=True,
    log_input_examples=False,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    save_model_kwargs=None,
    extra_tags=None,
):
    """
    Enable autologging for Keras.

    This method configures the autologging for Keras workflow. Only Keras > 3 is supported. For
    usage of lower Keras version (also known as tf-keras), please refer to `mlflow.tensorflow`
    flavor. At a high level, calling this `mlflow.keras.autolog()` function will replace
    `keras.Model.fit` method with the custom `fit` method provided by MLflow, which logs
    metrics/params/info/model to MLflow at the corresponding time.

    Autologging is compatible with all backends supported by Keras, including Tensorflow, PyTorch
    and JAX.

    Please note that autologging works only when you are using `model.fit()` for training. If you
    are writing a custom training loop, then you need to use manual logging.

    Args:
        log_every_epoch: If True, training metrics will be logged at the end of each epoch.
        log_every_n_steps: If set, training metrics will be logged every `n` training steps.
            `log_every_n_steps` must be `None` when `log_every_epoch=True`.
        log_models: If True, the Keras model will be logged to MLflow at the end of `model.fit()`.
        log_model_signatures: If True, model signature will be automatically captured and logged.
        save_exported_model: If True, model will be saved as the exported format (compiled graph),
            which is suitable for serving and deployment. If False, model will be saved in `.keras`
            format, which contains model architecture and weights.
        log_datasets: If True, the dataset metadata will be logged to MLflow.
        log_input_examples: If True, input examples will be logged.
        disable: If `True`, disables the Keras autologging.
        exclusive: If `True`, autologged content is not logged to user-created fluent runs. If
            `False`, autologged content is logged to the active fluent run, which may be
            user-created.  disable_for_unsupported_versions: If `True`, disable autologging for
            incompatible Keras versions.
        silent: If `True`, suppress all event logs and warnings from MLflow during Keras
            autologging.  If `True`, show all events and warnings during Keras autologging.
        registered_model_name: If set, each time a model is trained, it is registered as a new model
            version of the registered model with this name. The registered model is created if it
            does not already exist.
        save_model_kwargs: Extra kwargs passed to `keras.Model.save()`.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.

    .. code-block:: python
        :caption: Example

        import keras
        import mlflow
        import numpy as np

        mlflow.keras.autolog()

        # Prepare data for a 2-class classification.
        data = np.random.uniform([8, 28, 28, 3])
        label = np.random.randint(2, size=8)
        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(0.001),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        with mlflow.start_run() as run:
            model.fit(data, label, batch_size=4, epochs=2)
    """

    class FitPatch(PatchFunction):
        def __init__(self):
            pass

        def _infer_batch_size(self, inst, *args, **kwargs):
            batch_size = None
            if "batch_size" in kwargs:
                batch_size = kwargs["batch_size"]
            else:
                training_data = kwargs["x"] if "x" in kwargs else args[0]
                if _batch_size := getattr(training_data, "batch_size", None):
                    batch_size = _batch_size
                elif _batch_size := getattr(training_data, "_batch_size", None):
                    batch_size = (
                        _batch_size if isinstance(_batch_size, int) else _batch_size.numpy()
                    )
                elif is_iterator(training_data):
                    is_single_input_model = isinstance(inst.input_shape, tuple)
                    peek = next(training_data)
                    batch_size = len(peek[0]) if is_single_input_model else len(peek[0][0])

                    def origin_training_data_generator_fn():
                        yield peek
                        yield from training_data

                    origin_training_data = origin_training_data_generator_fn()

                    if "x" in kwargs:
                        kwargs["x"] = origin_training_data
                    else:
                        args = (origin_training_data,) + args[1:]
            return batch_size, args, kwargs

        def _patch_implementation(self, original, inst, *args, **kwargs):
            unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]

            batch_size, args, kwargs = self._infer_batch_size(inst, *args, **kwargs)

            if batch_size is not None:
                mlflow.log_param("batch_size", batch_size)
                unlogged_params.append("batch_size")

            log_fn_args_as_params(original, [], kwargs, unlogged_params)

            if log_datasets:
                try:
                    context_tags = context_registry.resolve_tags()
                    source = CodeDatasetSource(tags=context_tags)
                    x, y = _parse_dataset(*args, **kwargs)
                    _log_dataset(x, source, "train", targets=y)

                    if "validation_data" in kwargs:
                        _log_dataset(kwargs["validation_data"], source, "eval")

                except Exception as e:
                    _logger.warning(f"Failed to log dataset information to MLflow. Reason: {e}")

            # Add `MlflowCallback` to the callback list.
            callbacks = args[5] if len(args) >= 6 else kwargs.get("callbacks", [])
            mlflow_callback = MlflowCallback(
                log_every_epoch=log_every_epoch,
                log_every_n_steps=log_every_n_steps,
            )
            _check_existing_mlflow_callback(callbacks)
            callbacks.append(mlflow_callback)
            kwargs["callbacks"] = callbacks
            history = original(inst, *args, **kwargs)

            if log_models:
                _log_keras_model(inst, save_exported_model, log_model_signatures, save_model_kwargs)
            return history

    safe_patch("keras", keras.Model, "fit", FitPatch, manage_run=True, extra_tags=extra_tags)
