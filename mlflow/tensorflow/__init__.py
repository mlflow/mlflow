"""
The ``mlflow.tensorflow`` module provides an API for logging and loading TensorFlow models.
This module exports TensorFlow models with the following flavors:

TensorFlow (native) format
    This is the main flavor that can be loaded back into TensorFlow.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import os
import shutil
import logging
import concurrent.futures
import warnings
import atexit
import tempfile
from collections import namedtuple
import pandas
from packaging.version import Version
from threading import RLock
import numpy as np
import importlib
import yaml
import re

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.types.schema import TensorSpec
from mlflow.tracking.client import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import is_iterator
from mlflow.utils.environment import (
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
    _mlflow_conda_env,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _add_code_from_conf_to_system_path,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    resolve_input_example_and_signature,
    picklable_exception_safe_function,
    PatchFunction,
    log_fn_args_as_params,
    batch_metrics_logger,
    get_autologging_config,
)
from mlflow.utils.time_utils import get_current_time_millis
from mlflow.entities import Metric
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.context import registry as context_registry
from mlflow.models import infer_signature
from mlflow.exceptions import INVALID_PARAMETER_VALUE


FLAVOR_NAME = "tensorflow"

_logger = logging.getLogger(__name__)

_MAX_METRIC_QUEUE_SIZE = 500

_LOG_EVERY_N_STEPS = 1

_metric_queue_lock = RLock()
_metric_queue = []

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# For tracking if the run was started by autologging.
_AUTOLOG_RUN_ID = None

# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model"


_MODEL_TYPE_KERAS = "keras"
_MODEL_TYPE_TF1_ESTIMATOR = "tf1-estimator"
_MODEL_TYPE_TF2_MODULE = "tf2-module"


def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("tensorflow")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path,
    custom_objects=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    saved_model_kwargs=None,
    keras_model_kwargs=None,
    metadata=None,
):
    """
    Log a TF2 core model (inheriting tf.Module) or a Keras model in MLflow Model format.

    .. note::

        If you log a Keras or TensorFlow model without a signature, inference with
        :py:func:`mlflow.pyfunc.spark_udf()` will not work unless the model's pyfunc
        representation accepts pandas DataFrames as inference inputs.

        You can infer a model's signature by calling the :py:func:`mlflow.models.infer_signature()`
        API on features from the model's test dataset. You can also manually create a model
        signature, for example:

        .. code-block:: python
            :caption: Example of creating signature for saving TensorFlow and `tf.Keras` models

            from mlflow.types.schema import Schema, TensorSpec
            from mlflow.models.signature import ModelSignature
            import numpy as np

            input_schema = Schema(
                [
                    TensorSpec(np.dtype(np.uint64), (-1, 5), "field1"),
                    TensorSpec(np.dtype(np.float32), (-1, 3, 2), "field2"),
                ]
            )
            # Create the signature for a model that requires 2 inputs:
            #  - Input with name "field1", shape (-1, 5), type "np.uint64"
            #  - Input with name "field2", shape (-1, 3, 2), type "np.float32"
            signature = ModelSignature(inputs=input_schema)

    :param model: The TF2 core model (inheriting tf.Module) or Keras model to be saved.
    :param artifact_path: The run-relative path to which to log model artifacts.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.tensorflow.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
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
                        predictions = ...  # compute model predictions
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
    :param saved_model_kwargs: a dict of kwargs to pass to ``tensorflow.saved_model.save`` method.
    :param keras_model_kwargs: a dict of kwargs to pass to ``keras_model.save`` method.
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.tensorflow,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        custom_objects=custom_objects,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        saved_model_kwargs=saved_model_kwargs,
        keras_model_kwargs=keras_model_kwargs,
        metadata=metadata,
    )


def _save_keras_custom_objects(path, custom_objects):
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


_NO_MODEL_SIGNATURE_WARNING = (
    "You are saving a TensorFlow Core model or Keras model "
    "without a signature. Inference with mlflow.pyfunc.spark_udf() will not work "
    "unless the model's pyfunc representation accepts pandas DataFrames as "
    "inference inputs."
)


def _get_keras_version(keras_module):
    import tensorflow

    if Version(tensorflow.__version__) >= Version("2.6.0"):
        import keras

        return keras.__version__
    else:
        return keras_module.__version__


def save_model(
    model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    custom_objects=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    saved_model_kwargs=None,
    keras_model_kwargs=None,
    metadata=None,
):
    """
    Save a TF2 core model (inheriting tf.Module) or Keras model in MLflow Model format to a path on
    the local file system.

    .. note::
        If you save a Keras or TensorFlow model without a signature, inference with
        :py:func:`mlflow.pyfunc.spark_udf()` will not work unless the model's pyfunc
        representation accepts pandas DataFrames as inference inputs.
        You can infer a model's signature by calling the :py:func:`mlflow.models.infer_signature()`
        API on features from the model's test dataset. You can also manually create a model
        signature, for example:

        .. code-block:: python
            :caption: Example of creating signature for saving TensorFlow and `tf.Keras` models

            from mlflow.types.schema import Schema, TensorSpec
            from mlflow.models.signature import ModelSignature
            import numpy as np

            input_schema = Schema(
                [
                    TensorSpec(np.dtype(np.uint64), (-1, 5), "field1"),
                    TensorSpec(np.dtype(np.float32), (-1, 3, 2), "field2"),
                ]
            )
            # Create the signature for a model that requires 2 inputs:
            #  - Input with name "field1", shape (-1, 5), type "np.uint64"
            #  - Input with name "field2", shape (-1, 3, 2), type "np.float32"
            signature = ModelSignature(inputs=input_schema)

    :param model: The Keras model or Tensorflow module to be saved.
    :param path: Local path where the MLflow model is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: MLflow model configuration to which to add the ``tensorflow`` flavor.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.tensorflow.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param saved_model_kwargs: a dict of kwargs to pass to ``tensorflow.saved_model.save`` method
                               if the model to be saved is a Tensorflow module.
    :param keras_model_kwargs: a dict of kwargs to pass to ``model.save`` method if the model
                               to be saved is a keras model.
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import tensorflow
    from tensorflow.keras.models import Model as KerasModel

    if signature is None:
        _logger.warning(_NO_MODEL_SIGNATURE_WARNING)
    else:
        num_inputs = len(signature.inputs.inputs)
        if num_inputs == 0:
            raise MlflowException(
                "The model signature's input schema must contain at least one field.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        for field in signature.inputs.inputs:
            if not isinstance(field, TensorSpec):
                raise MlflowException(
                    "All fields in the model signature's input schema must be of type TensorSpec.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if field.shape[0] != -1:
                raise MlflowException(
                    "All fields in the model signature's input schema must have a shape "
                    "in which the first dimension is a variable dimension.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    # check if path exists
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    if isinstance(model, KerasModel):
        keras_model_kwargs = keras_model_kwargs or {}

        data_subpath = "data"
        # construct new data folder in existing path
        data_path = os.path.join(path, data_subpath)
        os.makedirs(data_path)
        model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)

        keras_module = importlib.import_module("tensorflow.keras")
        # save custom objects if there are custom objects
        if custom_objects is not None:
            _save_keras_custom_objects(data_path, custom_objects)

        # save keras module spec to path/data/keras_module.txt
        with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
            f.write(keras_module.__name__)

        # Use the SavedModel format if `save_format` is unspecified
        save_format = keras_model_kwargs.get("save_format", "tf")

        # save keras save_format to path/data/save_format.txt
        with open(os.path.join(data_path, _KERAS_SAVE_FORMAT_PATH), "w") as f:
            f.write(save_format)

        # save keras model
        # To maintain prior behavior, when the format is HDF5, we save
        # with the h5 file extension. Otherwise, model_path is a directory
        # where the saved_model.pb will be stored (for SavedModel format)
        file_extension = ".h5" if save_format == "h5" else ""
        model_path = os.path.join(path, model_subpath) + file_extension
        if path.startswith("/dbfs/"):
            # The Databricks Filesystem uses a FUSE implementation that does not support
            # random writes. It causes an error.
            with tempfile.NamedTemporaryFile(suffix=".h5") as f:
                model.save(f.name, **keras_model_kwargs)
                f.flush()  # force flush the data
                shutil.copyfile(src=f.name, dst=model_path)
        else:
            model.save(model_path, **keras_model_kwargs)

        pyfunc_options = {
            "data": data_subpath,
        }

        flavor_options = {
            **pyfunc_options,
            "model_type": _MODEL_TYPE_KERAS,
            "keras_version": _get_keras_version(keras_module),
            "save_format": save_format,
        }
    elif isinstance(model, tensorflow.Module):
        saved_model_kwargs = saved_model_kwargs or {}
        model_dir_subpath = "tf2model"
        model_path = os.path.join(path, model_dir_subpath)
        tensorflow.saved_model.save(model, model_path, **saved_model_kwargs)
        pyfunc_options = {}
        flavor_options = {
            "saved_model_dir": model_dir_subpath,
            "model_type": _MODEL_TYPE_TF2_MODULE,
        }
    else:
        raise MlflowException(f"Unknown model type: {type(model)}")

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(FLAVOR_NAME, code=code_dir_subpath, **flavor_options)

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.tensorflow",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **pyfunc_options,
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


def _load_keras_model(model_path, keras_module, save_format, **kwargs):
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
    unsuffixed_version = re.sub(r"-tf$", "", _get_keras_version(keras_module))
    if save_format == "h5" and Version(unsuffixed_version) >= Version("2.2.3"):
        # NOTE: Keras 2.2.3 does not work with unicode paths in python2. Pass in h5py.File instead
        # of string to avoid issues.
        import h5py

        with h5py.File(os.path.abspath(model_path), "r") as model_path:
            return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)
    else:
        # NOTE: Older versions of Keras only handle filepath.
        return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)


def _get_flavor_conf(model_conf):
    if "keras" in model_conf.flavors:
        return model_conf.flavors["keras"]
    return model_conf.flavors[FLAVOR_NAME]


def _infer_model_type(model_conf):
    model_type = _get_flavor_conf(model_conf).get("model_type")
    if model_type is not None:
        return model_type
    # Loading model logged by old version mlflow, which deos not record model_type
    # Inferring model type by checking whether model_conf contains "keras" flavor.
    if "keras" in model_conf.flavors:
        return _MODEL_TYPE_KERAS
    return _MODEL_TYPE_TF1_ESTIMATOR


def load_model(model_uri, dst_path=None, saved_model_kwargs=None, keras_model_kwargs=None):
    """
    Load an MLflow model that contains the TensorFlow flavor from the specified path.

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
    :param saved_model_kwargs: kwargs to pass to ``tensorflow.saved_model.load`` method.
                               Only available when you are loading a tensorflow2 core model.
    :param keras_model_kwargs: kwargs to pass to ``keras.models.load_model`` method.
                               Only available when you are loading a Keras model.

    :return: A callable graph (tf.function) that takes inputs and returns inferences.

    .. code-block:: python
        :caption: Example

        import mlflow
        import tensorflow as tf

        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            signature_definition = mlflow.tensorflow.load_model(
                model_uri="model_uri", tf_sess=tf_sess
            )
            input_tensors = [
                tf_graph.get_tensor_by_name(input_signature.name)
                for _, input_signature in signature_definition.inputs.items()
            ]
            output_tensors = [
                tf_graph.get_tensor_by_name(output_signature.name)
                for _, output_signature in signature_definition.outputs.items()
            ]
    """
    import tensorflow

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    model_configuration_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)
    model_conf = Model.load(model_configuration_path)

    flavor_conf = _get_flavor_conf(model_conf)

    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)

    model_type = _infer_model_type(model_conf)
    if model_type == _MODEL_TYPE_KERAS:
        keras_model_kwargs = keras_model_kwargs or {}
        keras_module = importlib.import_module(flavor_conf.get("keras_module", "tensorflow.keras"))
        # For backwards compatibility, we assume h5 when the save_format is absent
        save_format = flavor_conf.get("save_format", "h5")
        model_path = os.path.join(local_model_path, flavor_conf.get("data", _MODEL_SAVE_PATH))
        return _load_keras_model(
            model_path=model_path,
            keras_module=keras_module,
            save_format=save_format,
            **keras_model_kwargs,
        )
    if model_type == _MODEL_TYPE_TF1_ESTIMATOR:
        tf_saved_model_dir = os.path.join(local_model_path, flavor_conf["saved_model_dir"])
        tf_meta_graph_tags = flavor_conf["meta_graph_tags"]
        tf_signature_def_key = flavor_conf["signature_def_key"]
        return _load_tf1_estimator_saved_model(
            tf_saved_model_dir=tf_saved_model_dir,
            tf_meta_graph_tags=tf_meta_graph_tags,
            tf_signature_def_key=tf_signature_def_key,
        )
    if model_type == _MODEL_TYPE_TF2_MODULE:
        saved_model_kwargs = saved_model_kwargs or {}
        tf_saved_model_dir = os.path.join(local_model_path, flavor_conf["saved_model_dir"])
        return tensorflow.saved_model.load(tf_saved_model_dir, **saved_model_kwargs)

    raise MlflowException(f"Unknown model_type: {model_type}")


def _load_tf1_estimator_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Load a specified TensorFlow model consisting of a TensorFlow metagraph and signature definition
    from a serialized TensorFlow ``SavedModel`` collection.

    :param tf_saved_model_dir: The local filesystem path or run-relative artifact path to the model.
    :param tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the `tf.saved_model.builder.SavedModelBuilder
                               method <https://www.tensorflow.org/api_docs/python/tf/saved_model/
                               builder/SavedModelBuilder#add_meta_graph>`_.
    :param tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``SavedModel``'s
                                 signature definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.SavedModelBuilder`` method.
    :return: A callable graph (tensorflow.function) that takes inputs and returns inferences.
    """
    import tensorflow

    loaded = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
        tags=tf_meta_graph_tags, export_dir=tf_saved_model_dir
    )
    loaded_sig = loaded.signatures
    if tf_signature_def_key not in loaded_sig:
        raise MlflowException(
            "Could not find signature def key %s. Available keys are: %s"
            % (tf_signature_def_key, list(loaded_sig.keys()))
        )
    return loaded_sig[tf_signature_def_key]


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``. This function loads an MLflow
    model with the TensorFlow flavor into a new TensorFlow graph and exposes it behind the
    ``pyfunc.predict`` interface.

    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    """
    import tensorflow

    model_meta_path1 = os.path.join(path, MLMODEL_FILE_NAME)
    model_meta_path2 = os.path.join(os.path.dirname(path), MLMODEL_FILE_NAME)

    if os.path.isfile(model_meta_path1):
        model_meta = Model.load(model_meta_path1)
    elif os.path.isfile(model_meta_path2):
        model_meta = Model.load(model_meta_path2)
    else:
        raise MlflowException(f"Cannot find file {MLMODEL_FILE_NAME} for the logged model.")

    model_type = _infer_model_type(model_meta)
    if model_type == _MODEL_TYPE_KERAS:
        if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
            with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH)) as f:
                keras_module = importlib.import_module(f.read())
        else:
            import tensorflow.keras

            keras_module = tensorflow.keras

        # By default, we assume the save_format is h5 for backwards compatibility
        save_format = "h5"
        save_format_path = os.path.join(path, _KERAS_SAVE_FORMAT_PATH)
        if os.path.isfile(save_format_path):
            with open(save_format_path) as f:
                save_format = f.read()

        # In SavedModel format, if we don't compile the model
        should_compile = save_format == "tf"
        K = importlib.import_module(keras_module.__name__ + ".backend")
        if K.backend() == "tensorflow":
            K.set_learning_phase(0)
            m = _load_keras_model(
                path, keras_module=keras_module, save_format=save_format, compile=should_compile
            )
            return _KerasModelWrapper(m, model_meta.signature)
        else:
            raise MlflowException("Unsupported backend '%s'" % K._BACKEND)
    if model_type == _MODEL_TYPE_TF1_ESTIMATOR:
        flavor_conf = _get_flavor_configuration(path, FLAVOR_NAME)

        tf_saved_model_dir = os.path.join(path, flavor_conf["saved_model_dir"])
        tf_meta_graph_tags = flavor_conf["meta_graph_tags"]
        tf_signature_def_key = flavor_conf["signature_def_key"]

        loaded_model = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
            export_dir=tf_saved_model_dir, tags=tf_meta_graph_tags
        )
        return _TF2Wrapper(model=loaded_model, infer=loaded_model.signatures[tf_signature_def_key])
    if model_type == _MODEL_TYPE_TF2_MODULE:
        flavor_conf = _get_flavor_configuration(path, FLAVOR_NAME)
        tf_saved_model_dir = os.path.join(path, flavor_conf["saved_model_dir"])
        loaded_model = tensorflow.saved_model.load(tf_saved_model_dir)
        return _TF2ModuleWrapper(model=loaded_model, signature=model_meta.signature)

    raise MlflowException("Unknown model_type.")


class _TF2Wrapper:
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``. For TensorFlow versions >= 2.0.0.
    """

    def __init__(self, model, infer):
        """
        :param model: A Tensorflow SavedModel.
        :param infer: Tensorflow function returned by a saved model that is used for inference.
        """
        # Note: we need to retain the model reference in TF2Wrapper object, because the infer
        #  function in tensorflow will be `ConcreteFunction` which only retains WeakRefs to the
        #  variables they close over.
        #  See https://www.tensorflow.org/guide/function#deleting_tfvariables_between_function_calls
        self.model = model
        self.infer = infer

    def predict(self, data):
        import tensorflow

        feed_dict = {}
        if isinstance(data, dict):
            feed_dict = {k: tensorflow.constant(v) for k, v in data.items()}
        elif isinstance(data, pandas.DataFrame):
            for df_col_name in list(data):
                # If there are multiple columns with the same name, selecting the shared name
                # from the DataFrame will result in another DataFrame containing the columns
                # with the shared name. TensorFlow cannot make eager tensors out of pandas
                # DataFrames, so we convert the DataFrame to a numpy array here.
                val = data[df_col_name]
                if isinstance(val, pandas.DataFrame):
                    val = val.values
                else:
                    val = np.array(val.to_list())
                feed_dict[df_col_name] = tensorflow.constant(val)
        else:
            raise TypeError("Only dict and DataFrame input types are supported")

        raw_preds = self.infer(**feed_dict)
        pred_dict = {col_name: raw_preds[col_name].numpy() for col_name in raw_preds.keys()}
        for col in pred_dict.keys():
            # If the output tensor is not 1-dimensional
            # AND all elements have length of 1, flatten the array with `ravel()`
            if len(pred_dict[col].shape) != 1 and all(
                len(element) == 1 for element in pred_dict[col]
            ):
                pred_dict[col] = pred_dict[col].ravel()
            else:
                pred_dict[col] = pred_dict[col].tolist()

        if isinstance(data, dict):
            return pred_dict
        else:
            return pandas.DataFrame.from_dict(data=pred_dict)


class _TF2ModuleWrapper:
    def __init__(self, model, signature):
        self.model = model
        self.signature = signature

    def predict(self, data):
        import tensorflow

        if isinstance(data, (np.ndarray, list)):
            data = tensorflow.convert_to_tensor(data)
        else:
            raise MlflowException(
                f"Unsupported input data type: {type(data)}, the input data must be "
                "numpy array or a list."
            )
        result = self.model(data)
        if isinstance(result, tensorflow.Tensor):
            return result.numpy()
        return result


class _KerasModelWrapper:
    def __init__(self, keras_model, signature):
        self.keras_model = keras_model
        self.signature = signature

    def predict(self, data):
        if isinstance(data, pandas.DataFrame):
            # This line is for backwards compatibility:
            # If model signature is not None, when calling
            # `keras_pyfunc_model.predict(pandas_dataframe)`, `_enforce_schema` will convert
            # dataframe input into dict input, so in the case `_KerasModelWrapper.predict`
            # will receive a dict type input.
            # If model signature is None, `_enforce_schema` can do nothing, and if the input
            # is dataframe, `_KerasModelWrapper.predict` will receive a dataframe input,
            # we need to handle this case, to keep backwards compatibility.
            return pandas.DataFrame(self.keras_model.predict(data.values), index=data.index)

        supported_input_types = (np.ndarray, list, tuple, dict)
        if not isinstance(data, supported_input_types):
            raise MlflowException(
                f"Unsupported input data type: {type(data)}. "
                f"Must be one of: {[x.__name__ for x in supported_input_types]}",
                INVALID_PARAMETER_VALUE,
            )
        return self.keras_model.predict(data)


def _assoc_list_to_map(lst):
    """
    Convert an association list to a dictionary.
    """
    d = {}
    for run_id, metric in lst:
        d[run_id] = d[run_id] + [metric] if run_id in d else [metric]
    return d


def _flush_queue():
    """
    Flush the metric queue and log contents in batches to MLflow.
    Queue is divided into batches according to run id.
    """
    try:
        # Multiple queue flushes may be scheduled simultaneously on different threads
        # (e.g., if the queue is at its flush threshold and several more items
        # are added before a flush occurs). For correctness and efficiency, only one such
        # flush operation should proceed; all others are redundant and should be dropped
        acquired_lock = _metric_queue_lock.acquire(blocking=False)
        if acquired_lock:
            client = MlflowClient()
            # For thread safety and to avoid modifying a list while iterating over it, we record a
            # separate list of the items being flushed and remove each one from the metric queue,
            # rather than clearing the metric queue or reassigning it (clearing / reassigning is
            # dangerous because we don't block threads from adding to the queue while a flush is
            # in progress)
            snapshot = _metric_queue[:]
            for item in snapshot:
                _metric_queue.remove(item)

            metrics_by_run = _assoc_list_to_map(snapshot)
            for run_id, metrics in metrics_by_run.items():
                client.log_batch(run_id, metrics=metrics, params=[], tags=[])
    finally:
        if acquired_lock:
            _metric_queue_lock.release()


def _add_to_queue(key, value, step, time, run_id):
    """
    Add a metric to the metric queue. Flush the queue if it exceeds
    max size.
    """
    met = Metric(key=key, value=value, timestamp=time, step=step)
    _metric_queue.append((run_id, met))
    if len(_metric_queue) > _MAX_METRIC_QUEUE_SIZE:
        _thread_pool.submit(_flush_queue)


def _log_event(event):
    """
    Extracts metric information from the event protobuf
    """
    if event.WhichOneof("what") == "summary":
        summary = event.summary
        for v in summary.value:
            if v.HasField("simple_value"):
                # NB: Most TensorFlow APIs use one-indexing for epochs, while tf.Keras
                # uses zero-indexing. Accordingly, the modular arithmetic used here is slightly
                # different from the arithmetic used in `__MLflowTfKeras2Callback.on_epoch_end`,
                # which provides metric logging hooks for tf.Keras
                if (event.step - 1) % _LOG_EVERY_N_STEPS == 0:
                    _add_to_queue(
                        key=v.tag,
                        value=v.simple_value,
                        step=event.step,
                        time=get_current_time_millis(),
                        run_id=mlflow.active_run().info.run_id,
                    )


@picklable_exception_safe_function
def _get_tensorboard_callback(lst):
    import tensorflow

    for x in lst:
        if isinstance(x, tensorflow.keras.callbacks.TensorBoard):
            return x
    return None


# A representation of a TensorBoard event logging directory with two attributes:
# :location - string: The filesystem location of the logging directory
# :is_temp - boolean: `True` if the logging directory was created for temporary use by MLflow,
#                     `False` otherwise
_TensorBoardLogDir = namedtuple("_TensorBoardLogDir", ["location", "is_temp"])


def _setup_callbacks(lst, metrics_logger):
    """
    Adds TensorBoard and MlfLowTfKeras callbacks to the
    input list, and returns the new list and appropriate log directory.
    """
    # pylint: disable=no-name-in-module
    from mlflow.tensorflow._autolog import _TensorBoard, __MLflowTfKeras2Callback

    tb = _get_tensorboard_callback(lst)
    if tb is None:
        log_dir = _TensorBoardLogDir(location=tempfile.mkdtemp(), is_temp=True)

        out_list = lst + [_TensorBoard(log_dir.location)]
    else:
        log_dir = _TensorBoardLogDir(location=tb.log_dir, is_temp=False)
        out_list = lst
    out_list += [__MLflowTfKeras2Callback(metrics_logger, _LOG_EVERY_N_STEPS)]
    return out_list, log_dir


@autologging_integration(FLAVOR_NAME)
def autolog(
    every_n_iter=1,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    log_input_examples=False,
    log_model_signatures=True,
    saved_model_kwargs=None,
    keras_model_kwargs=None,
):  # pylint: disable=unused-argument
    # pylint: disable=no-name-in-module
    """
    Enables autologging for ``tf.keras`` and ``keras``.
    Note that only ``tensorflow>=2.3`` are supported.
    As an example, try running the
    `Keras/TensorFlow example <https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py>`_.

    For each TensorFlow module, autologging captures the following information:

    **tf.keras**
     - **Metrics** and **Parameters**

      - Training loss; validation loss; user-specified metrics
      - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon

     - **Artifacts**

      - Model summary on training start
      - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model)
      - TensorBoard logs on training end

    **tf.keras.callbacks.EarlyStopping**
     - **Metrics** and **Parameters**

      - Metrics from the ``EarlyStopping`` callbacks: ``stopped_epoch``, ``restored_epoch``,
        ``restore_best_weight``, etc
      - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``:
        ``min_delta``, ``patience``, ``baseline``, ``restore_best_weights``, etc

    Refer to the autologging tracking documentation for more
    information on `TensorFlow workflows
    <https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras-experimental>`_.

    :param every_n_iter: The frequency with which metrics should be logged. For example, a value of
                         100 will log metrics at step 0, 100, 200, etc.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
                         If ``False``, dataset information is not logged.
    :param disable: If ``True``, disables the TensorFlow autologging integration. If ``False``,
                    enables the TensorFlow integration autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      tensorflow that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during TensorFlow
                   autologging. If ``False``, show all events and warnings during TensorFlow
                   autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with tf/keras model artifacts during training. If
                               ``False``, input examples are not logged.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with tf/keras model artifacts during training. If ``False``,
                                 signatures are not logged. Note that logging TensorFlow models
                                 with signatures changes their pyfunc inference behavior when
                                 Pandas DataFrames are passed to ``predict()``.
                                 When a signature is present, an ``np.ndarray``
                                 (for single-output models) or a mapping from
                                 ``str`` -> ``np.ndarray`` (for multi-output models) is returned;
                                 when a signature is not present, a Pandas DataFrame is returned.
    :param saved_model_kwargs: a dict of kwargs to pass to ``tensorflow.saved_model.save`` method.
    :param keras_model_kwargs: a dict of kwargs to pass to ``keras_model.save`` method.
    """
    import tensorflow

    global _LOG_EVERY_N_STEPS
    _LOG_EVERY_N_STEPS = every_n_iter

    atexit.register(_flush_queue)

    if Version(tensorflow.__version__) < Version("2.3"):
        warnings.warn("Could not log to MLflow. TensorFlow versions below 2.3 are not supported.")
        return

    @picklable_exception_safe_function
    def _get_early_stop_callback(callbacks):
        for callback in callbacks:
            if isinstance(callback, tensorflow.keras.callbacks.EarlyStopping):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {
                    "monitor": callback.monitor,
                    "min_delta": callback.min_delta,
                    "patience": callback.patience,
                    "baseline": callback.baseline,
                    "restore_best_weights": callback.restore_best_weights,
                }
                mlflow.log_params(earlystopping_params)
            except Exception:
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:
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

    def _log_keras_model(history, args):
        def _infer_model_signature(input_data_slice):
            # In certain TensorFlow versions, calling `predict()` on model  may modify
            # the `stop_training` attribute, so we save and restore it accordingly
            original_stop_training = history.model.stop_training
            model_output = history.model.predict(input_data_slice)
            history.model.stop_training = original_stop_training
            return infer_signature(input_data_slice, model_output)

        from mlflow.tensorflow._autolog import extract_tf_keras_input_example

        def _get_tf_keras_input_example_slice():
            input_training_data = args[0]
            keras_input_example_slice = extract_tf_keras_input_example(input_training_data)
            if keras_input_example_slice is None:
                raise MlflowException(
                    "Cannot log input example or model signature for input with type"
                    f" {type(input_training_data)}. TensorFlow Keras autologging can"
                    " only log input examples and model signatures for the following"
                    " input types: numpy.ndarray, dict[string -> numpy.ndarray],"
                    " tensorflow.keras.utils.Sequence, and"
                    " tensorflow.data.Dataset (TensorFlow >= 2.1.0 required)",
                    INVALID_PARAMETER_VALUE,
                )
            return keras_input_example_slice

        input_example, signature = resolve_input_example_and_signature(
            _get_tf_keras_input_example_slice,
            _infer_model_signature,
            log_input_examples,
            log_model_signatures,
            _logger,
        )

        log_model(
            model=history.model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            registered_model_name=get_autologging_config(
                FLAVOR_NAME, "registered_model_name", None
            ),
            saved_model_kwargs=saved_model_kwargs,
            keras_model_kwargs=keras_model_kwargs,
        )

    class FitPatch(PatchFunction):
        def __init__(self):
            self.log_dir = None

        def _patch_implementation(
            self, original, inst, *args, **kwargs
        ):  # pylint: disable=arguments-differ
            unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]

            batch_size = None
            try:
                is_single_input_model = isinstance(inst.input_shape, tuple)
                training_data = kwargs["x"] if "x" in kwargs else args[0]
                if isinstance(training_data, tensorflow.data.Dataset) and hasattr(
                    training_data, "_batch_size"
                ):
                    batch_size = training_data._batch_size.numpy()
                elif isinstance(training_data, tensorflow.keras.utils.Sequence):
                    first_batch_inputs, _ = training_data[0]
                    if is_single_input_model:
                        batch_size = len(first_batch_inputs)
                    else:
                        batch_size = len(first_batch_inputs[0])
                elif is_iterator(training_data):
                    peek = next(training_data)
                    if is_single_input_model:
                        batch_size = len(peek[0])
                    else:
                        batch_size = len(peek[0][0])

                    def __restore_generator(prev_generator):
                        yield peek
                        yield from prev_generator

                    restored_generator = __restore_generator(training_data)
                    if "x" in kwargs:
                        kwargs["x"] = restored_generator
                    else:
                        args = (restored_generator,) + args[1:]
            except Exception as e:
                _logger.warning(
                    "Encountered unexpected error while inferring batch size from training"
                    " dataset: %s",
                    e,
                )

            if batch_size is not None:
                mlflow.log_param("batch_size", batch_size)
                unlogged_params.append("batch_size")

            log_fn_args_as_params(original, args, kwargs, unlogged_params)

            run_id = mlflow.active_run().info.run_id
            with batch_metrics_logger(run_id) as metrics_logger:
                # Check if the 'callback' argument of fit() is set positionally
                if len(args) >= 6:
                    # Convert the positional training function arguments to a list in order to
                    # mutate the contents
                    args = list(args)
                    # Make a shallow copy of the preexisting callbacks to avoid permanently
                    # modifying their contents for future training invocations. Introduce
                    # TensorBoard & tf.keras callbacks if necessary
                    callbacks = list(args[5])
                    callbacks, self.log_dir = _setup_callbacks(callbacks, metrics_logger)
                    # Replace the callbacks positional entry in the copied arguments and convert
                    # the arguments back to tuple form for usage in the training function
                    args[5] = callbacks
                    args = tuple(args)
                else:
                    # Make a shallow copy of the preexisting callbacks and introduce TensorBoard
                    # & tf.keras callbacks if necessary
                    callbacks = list(kwargs.get("callbacks") or [])
                    kwargs["callbacks"], self.log_dir = _setup_callbacks(callbacks, metrics_logger)

                early_stop_callback = _get_early_stop_callback(callbacks)
                _log_early_stop_callback_params(early_stop_callback)

                if log_datasets:
                    try:
                        context_tags = context_registry.resolve_tags()
                        source = CodeDatasetSource(tags=context_tags)

                        x = kwargs["x"] if "x" in kwargs else args[0]
                        if "y" in kwargs:
                            y = kwargs["y"]
                        elif len(args) >= 2:
                            y = args[1]
                        else:
                            y = None

                        if "validation_data" in kwargs:
                            validation_data = kwargs["validation_data"]
                        elif len(args) >= 8:
                            validation_data = args[7]
                        else:
                            validation_data = None
                        _log_tensorflow_dataset(x, source, "train", targets=y)
                        if validation_data is not None:
                            _log_tensorflow_dataset(validation_data, source, "eval")

                    except Exception as e:
                        _logger.warning(
                            "Failed to log training dataset information to "
                            "MLflow Tracking. Reason: %s",
                            e,
                        )

                history = original(inst, *args, **kwargs)

                if log_models:
                    _log_keras_model(history, args)

                _log_early_stop_callback_metrics(
                    callback=early_stop_callback,
                    history=history,
                    metrics_logger=metrics_logger,
                )

                _flush_queue()
                mlflow.log_artifacts(
                    local_dir=self.log_dir.location,
                    artifact_path="tensorboard_logs",
                )
            if self.log_dir.is_temp:
                shutil.rmtree(self.log_dir.location)
            return history

        def _on_exception(self, exception):
            if (
                self.log_dir is not None
                and self.log_dir.is_temp
                and os.path.exists(self.log_dir.location)
            ):
                shutil.rmtree(self.log_dir.location)

    managed = [
        (tensorflow.keras.Model, "fit", FitPatch),
    ]

    for p in managed:
        safe_patch(FLAVOR_NAME, *p, manage_run=True)


def _log_tensorflow_dataset(tensorflow_dataset, source, context, name=None, targets=None):
    import tensorflow

    # create a dataset
    if isinstance(tensorflow_dataset, np.ndarray):
        dataset = from_numpy(features=tensorflow_dataset, targets=targets, source=source, name=name)
    elif isinstance(tensorflow_dataset, tensorflow.Tensor):
        dataset = from_tensorflow(
            features=tensorflow_dataset, targets=targets, source=source, name=name
        )
    elif isinstance(tensorflow_dataset, tensorflow.data.Dataset):
        dataset = from_tensorflow(features=tensorflow_dataset, source=source, name=name)
    elif isinstance(tensorflow_dataset, tuple):
        x = tensorflow_dataset[0]
        y = tensorflow_dataset[1]
        # check if x and y are tensors
        if isinstance(x, tensorflow.Tensor) and isinstance(y, tensorflow.Tensor):
            dataset = from_tensorflow(features=x, source=source, targets=y, name=name)
        else:
            dataset = from_numpy(features=x, targets=y, source=source, name=name)
    else:
        _logger.warning(
            "Unrecognized dataset type %s. Dataset logging skipped.", type(tensorflow_dataset)
        )
        return

    mlflow.log_input(dataset, context)
