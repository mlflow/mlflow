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
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model"
_PIP_ENV_SUBPATH = "requirements.txt"


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
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

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
    if K.backend() == "tensorflow":
        K.set_learning_phase(0)
        m = _load_model(
            path, keras_module=keras_module, save_format=save_format, compile=should_compile
        )
        return _KerasModelWrapper(m, None, None)
    else:
        raise MlflowException("Unsupported backend '%s'" % K._BACKEND)


def load_model(local_model_path, flavor_conf, **kwargs):
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
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


def log_model(
    keras_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    custom_objects=None,
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
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
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
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.

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
    from mlflow.tensorflow import keras as keras_flavor
    if signature is not None:
        warnings.warn(
            "The pyfunc inference behavior of Keras models logged "
            "with signatures differs from the behavior of Keras "
            "models logged without signatures. Specifically, when a "
            "signature is present, passing a Pandas DataFrame as "
            "input to the pyfunc `predict()` API produces an `ndarray` "
            "(for single-output models) or a dictionary of `str -> ndarray`: "
            "(for multi-output models). In contrast, when a signature "
            "is *not* present, `predict()` produces "
            "a Pandas DataFrame output in response to a Pandas DataFrame input."
        )

    return Model.log(
        artifact_path=artifact_path,
        flavor=keras_flavor,
        keras_model=keras_model,
        conda_env=conda_env,
        code_paths=code_paths,
        custom_objects=custom_objects,
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


def save_model(
    keras_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    custom_objects=None,
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
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
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
    import keras.models
    from mlflow.tensorflow import get_default_pip_requirements, FLAVOR_NAME

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if not isinstance(keras_model, keras.models.Model):
        raise MlflowException(
            "The `keras_model` argument value is not a keras model instance."
        )
    keras_module = importlib.import_module("keras")

    # check if path exists
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    # construct new data folder in existing path
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

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
        code=code_dir_subpath,
    )

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.tensorflow.keras",
        data=data_subpath,
        env=_CONDA_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    include_cloudpickle = custom_objects is not None
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(include_cloudpickle)
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

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))
