"""
The ``mlflow.keras`` module provides an API for logging and loading Keras models. This module
exports Keras models with the following flavors:

Keras (native) format
    This is the main flavor that can be loaded back into Keras.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import importlib
import os
import re
import yaml
import tempfile
import shutil
import warnings

import pandas as pd

from packaging.version import Version
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    ExceptionSafeClass,
    log_fn_args_as_params,
    batch_metrics_logger,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


FLAVOR_NAME = "keras"
# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model"
_PIP_ENV_SUBPATH = "requirements.txt"


def get_default_pip_requirements(include_cloudpickle=False, keras_module=None):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import tensorflow as tf

    pip_deps = [_get_pinned_requirement("tensorflow")]

    keras_module = keras_module or __import__("keras")
    is_plain_keras = keras_module.__name__ == "keras"
    tf_version = Version(tf.__version__)
    if (
        is_plain_keras
        # tensorflow >= 2.6.0 requires keras:
        # https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/tools/pip_package/setup.py#L106
        # To prevent a different version of keras from being installed by tensorflow when creating
        # a serving environment, add a pinned requirement for keras
        or tf_version >= Version("2.6.0")
    ):
        pip_deps.append(_get_pinned_requirement("keras"))

    # Tensorflow<2.4 does not work with h5py>=3.0.0
    # see https://github.com/tensorflow/tensorflow/issues/44467
    if tf_version < Version("2.4"):
        pip_deps.append("h5py<3.0.0")

    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env(include_cloudpickle=False, keras_module=None):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(include_cloudpickle, keras_module)
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    keras_model,
    path,
    conda_env=None,
    mlflow_model=None,
    custom_objects=None,
    keras_module=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param keras_module: Keras module to be used to save / load the model
                         (``keras`` or ``tf.keras``). If not provided, MLflow will
                         attempt to infer the Keras module based on the given model.
    :param kwargs: kwargs to pass to ``keras_model.save`` method.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

    .. code-block:: python
        :caption: Example

        import mlflow
        # Build, compile, and train your model
        keras_model = ...
        keras_model_path = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
        # Save the model as an MLflow Model
        mlflow.keras.save_model(keras_model, keras_model_path)
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if keras_module is None:

        def _is_plain_keras(model):
            try:
                import keras

                # NB: Network is the first parent with save method
                import keras.engine.network

                return isinstance(model, keras.engine.network.Network)
            except ImportError:
                return False

        def _is_tf_keras(model):
            try:
                # NB: Network is not exposed in tf.keras, we check for Model instead.
                import tensorflow.keras.models

                return isinstance(model, tensorflow.keras.models.Model)
            except ImportError:
                return False

        if _is_plain_keras(keras_model):
            keras_module = importlib.import_module("keras")
        elif _is_tf_keras(keras_model):
            keras_module = importlib.import_module("tensorflow.keras")
        else:
            raise MlflowException(
                "Unable to infer keras module from the model, please specify "
                "which keras module ('keras' or 'tensorflow.keras') is to be "
                "used to save and load the model."
            )
    elif type(keras_module) == str:
        keras_module = importlib.import_module(keras_module)

    # check if path exists
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))

    # construct new data folder in existing path
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # save custom objects if there are custom objects
    if custom_objects is not None:
        _save_custom_objects(data_path, custom_objects)

    # save keras module spec to path/data/keras_module.txt
    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    # Use the SavedModel format if `save_format` is unspecified
    save_format = kwargs.get("save_format", "tf")

    # save keras save_format to path/data/save_format.txt
    with open(os.path.join(data_path, _KERAS_SAVE_FORMAT_PATH), "w") as f:
        f.write(save_format)

    # save keras model
    # To maintain prior behavior, when the format is HDF5, we save
    # with the h5 file extension. Otherwise, model_path is a directory
    # where the saved_model.pb will be stored (for SavedModel format)
    file_extension = ".h5" if save_format == "h5" else ""
    model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
    model_path = os.path.join(path, model_subpath) + file_extension
    if path.startswith("/dbfs/"):
        # The Databricks Filesystem uses a FUSE implementation that does not support
        # random writes. It causes an error.
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            keras_model.save(f.name, **kwargs)
            f.flush()  # force flush the data
            shutil.copyfile(src=f.name, dst=model_path)
    else:
        keras_model.save(model_path, **kwargs)

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        keras_module=keras_module.__name__,
        keras_version=keras_module.__version__,
        save_format=save_format,
        data=data_subpath,
    )

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model, loader_module="mlflow.keras", data=data_subpath, env=_CONDA_ENV_FILE_NAME
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    include_cloudpickle = custom_objects is not None
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(include_cloudpickle, keras_module)
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    keras_model,
    artifact_path,
    conda_env=None,
    custom_objects=None,
    keras_module=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a Keras model as an MLflow artifact for the current run.

    :param keras_model: Keras model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param keras_module: Keras module to be used to save / load the model
                         (``keras`` or ``tf.keras``). If not provided, MLflow will
                         attempt to infer the Keras module based on the given model.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``keras_model.save`` method.

    .. code-block:: python
        :caption: Example

        from keras import Dense, layers
        import mlflow
        # Build, compile, and train your model
        keras_model = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
        # Log metrics and log the model
        with mlflow.start_run() as run:
            mlflow.keras.log_model(keras_model, "models")
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.keras,
        keras_model=keras_model,
        conda_env=conda_env,
        custom_objects=custom_objects,
        keras_module=keras_module,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def _save_custom_objects(path, custom_objects):
    """
    Save custom objects dictionary to a cloudpickle file so a model can be easily loaded later.

    :param path: An absolute path that points to the data directory within /path/to/model.
    :param custom_objects: Keras ``custom_objects`` is a dictionary mapping
                           names (strings) to custom classes or functions to be considered
                           during deserialization. MLflow saves these custom layers using
                           CloudPickle and restores them automatically when the model is
                           loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    """
    import cloudpickle

    custom_objects_path = os.path.join(path, _CUSTOM_OBJECTS_SAVE_PATH)
    with open(custom_objects_path, "wb") as out_f:
        cloudpickle.dump(custom_objects, out_f)


def _load_model(model_path, keras_module, save_format, **kwargs):
    keras_models = importlib.import_module(keras_module.__name__ + ".models")
    custom_objects = kwargs.pop("custom_objects", {})
    custom_objects_path = None
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)):
            custom_objects_path = os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if custom_objects_path is not None:
        import cloudpickle

        with open(custom_objects_path, "rb") as in_f:
            pickled_custom_objects = cloudpickle.load(in_f)
            pickled_custom_objects.update(custom_objects)
            custom_objects = pickled_custom_objects

    # If the save_format is HDF5, then we save with h5 file
    # extension to align with prior behavior of mlflow logging
    if save_format == "h5":
        model_path = model_path + ".h5"

    # keras in tensorflow used to have a '-tf' suffix in the version:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.1/tensorflow/python/keras/__init__.py#L36
    unsuffixed_version = re.sub(r"-tf$", "", keras_module.__version__)
    if save_format == "h5" and Version(unsuffixed_version) >= Version("2.2.3"):
        # NOTE: Keras 2.2.3 does not work with unicode paths in python2. Pass in h5py.File instead
        # of string to avoid issues.
        import h5py

        with h5py.File(os.path.abspath(model_path), "r") as model_path:
            return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)
    else:
        # NOTE: Older versions of Keras only handle filepath.
        return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)


class _KerasModelWrapper:
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self._graph = graph
        self._sess = sess

    def predict(self, data):
        def _predict(data):
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(self.keras_model.predict(data.values))
                predicted.index = data.index
            else:
                predicted = self.keras_model.predict(data)
            return predicted

        # In TensorFlow < 2.0, we use a graph and session to predict
        if self._graph is not None:
            with self._graph.as_default():
                with self._sess.as_default():
                    predicted = _predict(data)
        # In TensorFlow >= 2.0, we do not use a graph and session to predict
        else:
            predicted = _predict(data)
        return predicted


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``keras`` flavor.
    """
    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import keras

        keras_module = keras

    # By default, we assume the save_format is h5 for backwards compatibility
    save_format = "h5"
    save_format_path = os.path.join(path, _KERAS_SAVE_FORMAT_PATH)
    if os.path.isfile(save_format_path):
        with open(save_format_path, "r") as f:
            save_format = f.read()

    # In SavedModel format, if we don't compile the model
    should_compile = save_format == "tf"
    K = importlib.import_module(keras_module.__name__ + ".backend")
    if keras_module.__name__ == "tensorflow.keras" or K.backend() == "tensorflow":
        K.set_learning_phase(0)
        m = _load_model(
            path, keras_module=keras_module, save_format=save_format, compile=should_compile
        )
        return _KerasModelWrapper(m, None, None)

    else:
        raise MlflowException("Unsupported backend '%s'" % K._BACKEND)


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a Keras model from a local file or a run.

    Extra arguments are passed through to keras.load_model.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A Keras model instance.

    .. code-block:: python
        :caption: Example

        # Load persisted model as a Keras model or as a PyFunc, call predict() on a pandas DataFrame
        keras_model = mlflow.keras.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2" + "/models")
        predictions = keras_model.predict(x_test)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    keras_module = importlib.import_module(flavor_conf.get("keras_module", "keras"))
    keras_model_artifacts_path = os.path.join(
        local_model_path, flavor_conf.get("data", _MODEL_SAVE_PATH)
    )
    # For backwards compatibility, we assume h5 when the save_format is absent
    save_format = flavor_conf.get("save_format", "h5")
    return _load_model(
        model_path=keras_model_artifacts_path,
        keras_module=keras_module,
        save_format=save_format,
        **kwargs,
    )


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    # pylint: disable=E0611
    """
    Enables (or disables) and configures autologging from Keras to MLflow. Autologging captures
    the following information:

    **Metrics** and **Parameters**
     - Training loss; validation loss; user-specified metrics
     - Metrics associated with the ``EarlyStopping`` callbacks: ``stopped_epoch``,
       ``restored_epoch``, ``restore_best_weight``, ``last_epoch``, etc
     - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon
     - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``: ``min_delta``,
       ``patience``, ``baseline``, ``restore_best_weights``, etc
    **Artifacts**
     - Model summary on training start
     - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model) on training end

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.keras
        # Build, compile, enable autologging, and train your model
        keras_model = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        # autolog your metrics, parameters, and model
        mlflow.keras.autolog()
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))

    ``EarlyStopping Integration with Keras AutoLogging``

    MLflow will detect if an ``EarlyStopping`` callback is used in a ``fit()`` or
    ``fit_generator()`` call, and if the ``restore_best_weights`` parameter is set to be ``True``,
    then MLflow will log the metrics associated with the restored model as a final, extra step.
    The epoch of the restored model will also be logged as the metric ``restored_epoch``.
    This allows for easy comparison between the actual metrics of the restored model and
    the metrics of other models.

    If ``restore_best_weights`` is set to be ``False``, then MLflow will not log an additional step.

    Regardless of ``restore_best_weights``, MLflow will also log ``stopped_epoch``,
    which indicates the epoch at which training stopped due to early stopping.

    If training does not end due to early stopping, then ``stopped_epoch`` will be logged as ``0``.

    MLflow will also log the parameters of the ``EarlyStopping`` callback,
    excluding ``mode`` and ``verbose``.

    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the Keras autologging integration. If ``False``,
                    enables the Keras autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      keras that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Keras
                   autologging. If ``False``, show all events and warnings during Keras
                   autologging.
    """
    import keras

    if Version(keras.__version__) >= Version("2.6.0"):
        warnings.warn(
            (
                "Autologging support for keras >= 2.6.0 has been deprecated and will be removed in "
                "a future MLflow release. Use `mlflow.tensorflow.autolog()` instead."
            ),
            FutureWarning,
            stacklevel=2,
        )

    def getKerasCallback(metrics_logger):
        class __MLflowKerasCallback(keras.callbacks.Callback, metaclass=ExceptionSafeClass):
            """
            Callback for auto-logging metrics and parameters.
            Records available logs after each epoch.
            Records model structural information as params when training begins
            """

            def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
                mlflow.log_param("num_layers", len(self.model.layers))
                mlflow.log_param("optimizer_name", type(self.model.optimizer).__name__)
                if hasattr(self.model.optimizer, "lr"):
                    lr = (
                        self.model.optimizer.lr
                        if type(self.model.optimizer.lr) is float
                        else keras.backend.eval(self.model.optimizer.lr)
                    )
                    mlflow.log_param("learning_rate", lr)
                if hasattr(self.model.optimizer, "epsilon"):
                    epsilon = (
                        self.model.optimizer.epsilon
                        if type(self.model.optimizer.epsilon) is float
                        else keras.backend.eval(self.model.optimizer.epsilon)
                    )
                    mlflow.log_param("epsilon", epsilon)

                sum_list = []
                self.model.summary(print_fn=sum_list.append)
                summary = "\n".join(sum_list)
                tempdir = tempfile.mkdtemp()
                try:
                    summary_file = os.path.join(tempdir, "model_summary.txt")
                    with open(summary_file, "w") as f:
                        f.write(summary)
                    mlflow.log_artifact(local_path=summary_file)
                finally:
                    shutil.rmtree(tempdir)

            def on_epoch_end(self, epoch, logs=None):
                if not logs:
                    return
                metrics_logger.record_metrics(logs, epoch)

            def on_train_end(self, logs=None):
                if log_models:
                    log_model(self.model, artifact_path="model")

            # As of Keras 2.4.0, Keras Callback implementations must define the following
            # methods indicating whether or not the callback overrides functions for
            # batch training/testing/inference
            def _implements_train_batch_hooks(self):
                return False

            def _implements_test_batch_hooks(self):
                return False

            def _implements_predict_batch_hooks(self):
                return False

        return __MLflowKerasCallback()

    def _early_stop_check(callbacks):
        if Version(keras.__version__) >= Version("2.4.0"):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            earlystopping_params = {
                "monitor": callback.monitor,
                "min_delta": callback.min_delta,
                "patience": callback.patience,
                "baseline": callback.baseline,
                "restore_best_weights": callback.restore_best_weights,
            }
            mlflow.log_params(earlystopping_params)

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history, metrics_logger):
        if callback is None or not callback.model.stop_training:
            return

        callback_attrs = _get_early_stop_callback_attrs(callback)
        if callback_attrs is None:
            return

        stopped_epoch, restore_best_weights, _ = callback_attrs
        metrics_logger.record_metrics({"stopped_epoch": stopped_epoch})

        if not restore_best_weights or callback.best_weights is None:
            return

        monitored_metric = history.history.get(callback.monitor)
        if not monitored_metric:
            return

        initial_epoch = history.epoch[0]
        # If `monitored_metric` contains multiple best values (e.g. [0.1, 0.1, 0.2] where 0.1 is
        # the minimum loss), the epoch corresponding to the first occurrence of the best value is
        # the best epoch. In keras > 2.6.0, the best epoch can be obtained via the `best_epoch`
        # attribute of an `EarlyStopping` instance: https://github.com/keras-team/keras/pull/15197
        restored_epoch = initial_epoch + monitored_metric.index(callback.best)
        metrics_logger.record_metrics({"restored_epoch": restored_epoch})
        restored_index = history.epoch.index(restored_epoch)
        restored_metrics = {
            key: metrics[restored_index] for key, metrics in history.history.items()
        }
        # Checking that a metric history exists
        metric_key = next(iter(history.history), None)
        if metric_key is not None:
            metrics_logger.record_metrics(restored_metrics, stopped_epoch + 1)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            mlflowKerasCallback = getKerasCallback(metrics_logger)
            if len(args) > callback_arg_index:
                tmp_list = list(args)
                early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
                tmp_list[callback_arg_index] += [mlflowKerasCallback]
                args = tuple(tmp_list)
            elif kwargs.get("callbacks"):
                early_stop_callback = _early_stop_check(kwargs["callbacks"])
                kwargs["callbacks"] += [mlflowKerasCallback]
            else:
                kwargs["callbacks"] = [mlflowKerasCallback]

            _log_early_stop_callback_params(early_stop_callback)

            history = original(self, *args, **kwargs)

            _log_early_stop_callback_metrics(early_stop_callback, history, metrics_logger)

        return history

    def fit(original, self, *args, **kwargs):
        unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    def fit_generator(original, self, *args, **kwargs):
        """
        NOTE: `fit_generator()` is deprecated in Keras >= 2.4.0 and simply wraps `fit()`.
        To avoid unintentional creation of nested MLflow runs caused by a patched
        `fit_generator()` method calling a patched `fit()` method, we only patch
        `fit_generator()` in Keras < 2.4.0.
        """
        unlogged_params = ["self", "generator", "callbacks", "validation_data", "verbose"]
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    safe_patch(FLAVOR_NAME, keras.Model, "fit", fit, manage_run=True)
    # `fit_generator()` is deprecated in Keras >= 2.4.0 and simply wraps `fit()`.
    # To avoid unintentional creation of nested MLflow runs caused by a patched
    # `fit_generator()` method calling a patched `fit()` method, we only patch
    # `fit_generator()` in Keras < 2.4.0.
    if Version(keras.__version__) < Version("2.4.0"):
        safe_patch(FLAVOR_NAME, keras.Model, "fit_generator", fit_generator, manage_run=True)
