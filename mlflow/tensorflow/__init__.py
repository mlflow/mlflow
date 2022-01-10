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
import yaml
import logging
import concurrent.futures
import warnings
import atexit
import time
import tempfile
from collections import namedtuple
import pandas
from packaging.version import Version
from threading import RLock


import mlflow
import mlflow.keras
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME, _LOG_MODEL_METADATA_WARNING_TEMPLATE
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, get_artifact_uri
from mlflow.utils.annotations import keyword_only
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
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import _copy_file_or_tree, TempDir, write_to
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    picklable_exception_safe_function,
    PatchFunction,
    log_fn_args_as_params,
    batch_metrics_logger,
)
from mlflow.entities import Metric
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


FLAVOR_NAME = "tensorflow"

_logger = logging.getLogger(__name__)

_MAX_METRIC_QUEUE_SIZE = 500

_LOG_EVERY_N_STEPS = 1

_metric_queue_lock = RLock()
_metric_queue = []

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# For tracking if the run was started by autologging.
_AUTOLOG_RUN_ID = None


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import tensorflow as tf

    pip_deps = [_get_pinned_requirement("tensorflow")]

    # tensorflow >= 2.6.0 requires keras:
    # https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/tools/pip_package/setup.py#L106
    # To prevent a different version of keras from being installed by tensorflow when creating
    # a serving environment, add a pinned requirement for keras
    if Version(tf.__version__) >= Version("2.6.0"):
        pip_deps.append(_get_pinned_requirement("keras"))

    return pip_deps


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    tf_saved_model_dir,
    tf_meta_graph_tags,
    tf_signature_def_key,
    artifact_path,
    conda_env=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a *serialized* collection of TensorFlow graphs and variables as an MLflow model
    for the current run. This method operates on TensorFlow variables and graphs that have been
    serialized in TensorFlow's ``SavedModel`` format. For more information about ``SavedModel``
    format, see the TensorFlow documentation:
    https://www.tensorflow.org/guide/saved_model#save_and_restore_models.

    This method saves a model with both ``python_function`` and ``tensorflow`` flavors.
    If loaded back using the ``python_function`` flavor, the model can be used to predict on
    pandas DataFrames, producing a pandas DataFrame whose output columns correspond to the
    TensorFlow model's outputs. The python_function model will flatten outputs that are length-one,
    one-dimensional tensors of a single scalar value (e.g.
    ``{"predictions": [[1.0], [2.0], [3.0]]}``) into the scalar values (e.g.
    ``{"predictions": [1, 2, 3]}``), so that the resulting output column is a column of scalars
    rather than lists of length one. All other model output types are included as-is in the output
    DataFrame.

    :param tf_saved_model_dir: Path to the directory containing serialized TensorFlow variables and
                               graphs in ``SavedModel`` format.
    :param tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the
                               ``tf.saved_model.builder.SavedModelBuilder`` method.
    :param tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``SavedModel`` signature
                                 definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.SavedModelBuilder`` method.
    :param artifact_path: The run-relative path to which to log model artifacts.
    :param conda_env: {{ conda_env }}
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
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.tensorflow,
        tf_saved_model_dir=tf_saved_model_dir,
        tf_meta_graph_tags=tf_meta_graph_tags,
        tf_signature_def_key=tf_signature_def_key,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    tf_saved_model_dir,
    tf_meta_graph_tags,
    tf_signature_def_key,
    path,
    mlflow_model=None,
    conda_env=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a *serialized* collection of TensorFlow graphs and variables as an MLflow model
    to a local path. This method operates on TensorFlow variables and graphs that have been
    serialized in TensorFlow's ``SavedModel`` format. For more information about ``SavedModel``
    format, see the TensorFlow documentation:
    https://www.tensorflow.org/guide/saved_model#save_and_restore_models.

    :param tf_saved_model_dir: Path to the directory containing serialized TensorFlow variables and
                               graphs in ``SavedModel`` format.
    :param tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the
                               ``tf.saved_model.builder.savedmodelbuilder`` method.
    :param tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``savedmodel``
                                 signature definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.savedmodelbuilder`` method.
    :param path: Local path where the MLflow model is to be saved.
    :param mlflow_model: MLflow model configuration to which to add the ``tensorflow`` flavor.
    :param conda_env: {{ conda_env }}
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
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    _logger.info(
        "Validating the specified TensorFlow model by attempting to load it in a new TensorFlow"
        " graph..."
    )
    _validate_saved_model(
        tf_saved_model_dir=tf_saved_model_dir,
        tf_meta_graph_tags=tf_meta_graph_tags,
        tf_signature_def_key=tf_signature_def_key,
    )
    _logger.info("Validation succeeded!")

    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path), DIRECTORY_NOT_EMPTY)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    root_relative_path = _copy_file_or_tree(src=tf_saved_model_dir, dst=path, dst_dir=None)
    model_dir_subpath = "tfmodel"
    model_dir_path = os.path.join(path, model_dir_subpath)
    shutil.move(os.path.join(path, root_relative_path), model_dir_path)

    flavor_conf = dict(
        saved_model_dir=model_dir_subpath,
        meta_graph_tags=tf_meta_graph_tags,
        signature_def_key=tf_signature_def_key,
    )
    mlflow_model.add_flavor(FLAVOR_NAME, **flavor_conf)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", env=_CONDA_ENV_FILE_NAME)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
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


def _validate_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Validate the TensorFlow SavedModel by attempting to load it in a new TensorFlow graph.
    If the loading process fails, any exceptions thrown by TensorFlow are propagated.
    """
    _load_tensorflow_saved_model(
        tf_saved_model_dir=tf_saved_model_dir,
        tf_meta_graph_tags=tf_meta_graph_tags,
        tf_signature_def_key=tf_signature_def_key,
    )


def load_model(model_uri, dst_path=None):
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

    :return: A callable graph (tf.function) that takes inputs and returns inferences.

    .. code-block:: python
        :caption: Example

        import mlflow.tensorflow
        import tensorflow as tf
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            signature_definition = mlflow.tensorflow.load_model(model_uri="model_uri",
                                    tf_sess=tf_sess)
            input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                                for _, input_signature in signature_definition.inputs.items()]
            output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                                for _, output_signature in signature_definition.outputs.items()]
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    (
        tf_saved_model_dir,
        tf_meta_graph_tags,
        tf_signature_def_key,
    ) = _get_and_parse_flavor_configuration(model_path=local_model_path)
    return _load_tensorflow_saved_model(
        tf_saved_model_dir=tf_saved_model_dir,
        tf_meta_graph_tags=tf_meta_graph_tags,
        tf_signature_def_key=tf_signature_def_key,
    )


def _load_tensorflow_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
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


def _get_and_parse_flavor_configuration(model_path):
    """
    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    :return: A triple containing the following elements:

             - ``tf_saved_model_dir``: The local filesystem path to the underlying TensorFlow
                                       SavedModel directory.
             - ``tf_meta_graph_tags``: A list of tags identifying the TensorFlow model's metagraph
                                       within the serialized ``SavedModel`` object.
             - ``tf_signature_def_key``: A string identifying the input/output signature associated
                                         with the model. This is a key within the serialized
                                         ``SavedModel``'s signature definition mapping.
    """
    flavor_conf = _get_flavor_configuration(model_path=model_path, flavor_name=FLAVOR_NAME)
    tf_saved_model_dir = os.path.join(model_path, flavor_conf["saved_model_dir"])
    tf_meta_graph_tags = flavor_conf["meta_graph_tags"]
    tf_signature_def_key = flavor_conf["signature_def_key"]
    return tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``. This function loads an MLflow
    model with the TensorFlow flavor into a new TensorFlow graph and exposes it behind the
    ``pyfunc.predict`` interface.

    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    """
    import tensorflow

    (
        tf_saved_model_dir,
        tf_meta_graph_tags,
        tf_signature_def_key,
    ) = _get_and_parse_flavor_configuration(model_path=path)

    loaded_model = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
        export_dir=tf_saved_model_dir, tags=tf_meta_graph_tags
    )
    return _TF2Wrapper(model=loaded_model, infer=loaded_model.signatures[tf_signature_def_key])


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
                feed_dict[df_col_name] = tensorflow.constant(val)
        else:
            raise TypeError("Only dict and DataFrame input types are supported")

        raw_preds = self.infer(**feed_dict)
        pred_dict = {col_name: raw_preds[col_name].numpy() for col_name in raw_preds.keys()}
        for col in pred_dict.keys():
            if all(len(element) == 1 for element in pred_dict[col]):
                pred_dict[col] = pred_dict[col].ravel()
            else:
                pred_dict[col] = pred_dict[col].tolist()

        if isinstance(data, dict):
            return pred_dict
        else:
            return pandas.DataFrame.from_dict(data=pred_dict)


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
            client = mlflow.tracking.MlflowClient()
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
                        time=int(time.time() * 1000),
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


def _setup_callbacks(lst, log_models, metrics_logger):
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
    out_list += [__MLflowTfKeras2Callback(log_models, metrics_logger, _LOG_EVERY_N_STEPS)]
    return out_list, log_dir


@autologging_integration(FLAVOR_NAME)
def autolog(
    every_n_iter=1,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    # pylint: disable=E0611
    """
    Enables automatic logging from TensorFlow to MLflow.
    Note that autologging for ``tf.keras`` is handled by :py:func:`mlflow.tensorflow.autolog`,
    not :py:func:`mlflow.keras.autolog`.
    As an example, try running the
    `TensorFlow examples <https://github.com/mlflow/mlflow/tree/master/examples/tensorflow>`_.

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

    **tf.estimator**
     - **Metrics** and **Parameters**

      - TensorBoard metrics: ``average_loss``, ``loss``, etc
      - Parameters ``steps`` and ``max_steps``

     - **Artifacts**

      - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (TF saved model) on call
        to ``tf.estimator.export_saved_model``

    **TensorFlow Core**
     - **Metrics**

      - All ``tf.summary.scalar`` calls

    Refer to the autologging tracking documentation for more
    information on `TensorFlow workflows
    <https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras-experimental>`_.

    :param every_n_iter: The frequency with which metrics should be logged. For example, a value of
                         100 will log metrics at step 0, 100, 200, etc.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
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
    """
    import tensorflow

    global _LOG_EVERY_N_STEPS
    _LOG_EVERY_N_STEPS = every_n_iter

    atexit.register(_flush_queue)

    if Version(tensorflow.__version__) < Version("1.12"):
        warnings.warn("Could not log to MLflow. TensorFlow versions below 1.12 are not supported.")
        return

    try:
        from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
        from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.summary.writer.writer import FileWriter
    except ImportError:
        warnings.warn("Could not log to MLflow. TensorFlow versions below 1.12 are not supported.")
        return

    def train(original, self, *args, **kwargs):
        active_run = mlflow.active_run()
        global _AUTOLOG_RUN_ID
        _AUTOLOG_RUN_ID = active_run.info.run_id

        # Checking step and max_step parameters for logging
        if len(args) >= 3:
            mlflow.log_param("steps", args[2])
            if len(args) >= 4:
                mlflow.log_param("max_steps", args[3])
        if "steps" in kwargs:
            mlflow.log_param("steps", kwargs["steps"])
        if "max_steps" in kwargs:
            mlflow.log_param("max_steps", kwargs["max_steps"])

        result = original(self, *args, **kwargs)

        # Flush the metrics queue after training completes
        _flush_queue()

        # Log Tensorboard event files as artifacts
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if "tfevents" not in file:
                    continue
                mlflow.log_artifact(
                    local_path=os.path.join(self.model_dir, file),
                    artifact_path="tensorboard_logs",
                )
        return result

    def export_saved_model(original, self, *args, **kwargs):
        global _AUTOLOG_RUN_ID
        if _AUTOLOG_RUN_ID:
            _logger.info(
                "Logging TensorFlow Estimator as MLflow Model to run with ID '%s'", _AUTOLOG_RUN_ID
            )

            serialized = original(self, *args, **kwargs)

            def log_model_without_starting_new_run():
                """
                Performs the exact same operations as `log_model` without starting a new run
                """
                with TempDir() as tmp:
                    artifact_path = "model"
                    local_path = tmp.path("model")
                    mlflow_model = Model(artifact_path=artifact_path, run_id=_AUTOLOG_RUN_ID)
                    save_model_kwargs = dict(
                        tf_saved_model_dir=serialized.decode("utf-8"),
                        tf_meta_graph_tags=[tag_constants.SERVING],
                        tf_signature_def_key="predict",
                    )
                    save_model(path=local_path, mlflow_model=mlflow_model, **save_model_kwargs)
                    client = MlflowClient()
                    client.log_artifacts(_AUTOLOG_RUN_ID, local_path, artifact_path)

                    try:
                        client._record_logged_model(_AUTOLOG_RUN_ID, mlflow_model)
                    except MlflowException:
                        # We need to swallow all mlflow exceptions to maintain backwards
                        # compatibility with older tracking servers. Only print out a warning
                        # for now.
                        _logger.warning(
                            _LOG_MODEL_METADATA_WARNING_TEMPLATE,
                            get_artifact_uri(_AUTOLOG_RUN_ID),
                        )

            log_model_without_starting_new_run()

            _AUTOLOG_RUN_ID = None

        return serialized

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
            except Exception:  # pylint: disable=W0703
                return

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

    class FitPatch(PatchFunction):
        def __init__(self):
            self.log_dir = None

        def _patch_implementation(
            self, original, inst, *args, **kwargs
        ):  # pylint: disable=arguments-differ
            unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]

            log_fn_args_as_params(original, args, kwargs, unlogged_params)
            early_stop_callback = None

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
                    callbacks, self.log_dir = _setup_callbacks(
                        callbacks, log_models, metrics_logger
                    )
                    # Replace the callbacks positional entry in the copied arguments and convert
                    # the arguments back to tuple form for usage in the training function
                    args[5] = callbacks
                    args = tuple(args)
                else:
                    # Make a shallow copy of the preexisting callbacks and introduce TensorBoard
                    # & tf.keras callbacks if necessary
                    callbacks = list(kwargs.get("callbacks") or [])
                    kwargs["callbacks"], self.log_dir = _setup_callbacks(
                        callbacks, log_models, metrics_logger
                    )

                early_stop_callback = _get_early_stop_callback(callbacks)
                _log_early_stop_callback_params(early_stop_callback)

                history = original(inst, *args, **kwargs)

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

    class FitGeneratorPatch(PatchFunction):
        """
        NOTE: `fit_generator()` is deprecated in TF >= 2.1.0 and simply wraps `fit()`.
        To avoid unintentional creation of nested MLflow runs caused by a patched
        `fit_generator()` method calling a patched `fit()` method, we only patch
        `fit_generator()` in TF < 2.1.0.
        """

        def __init__(self):
            self.log_dir = None

        def _patch_implementation(
            self, original, inst, *args, **kwargs
        ):  # pylint: disable=arguments-differ
            unlogged_params = ["self", "generator", "callbacks", "validation_data", "verbose"]

            log_fn_args_as_params(original, args, kwargs, unlogged_params)

            run_id = mlflow.active_run().info.run_id

            with batch_metrics_logger(run_id) as metrics_logger:
                # Check if the 'callback' argument of fit() is set positionally
                if len(args) >= 5:
                    # Convert the positional training function arguments to a list in order to
                    # mutate the contents
                    args = list(args)
                    # Make a shallow copy of the preexisting callbacks to avoid permanently
                    # modifying their contents for future training invocations. Introduce
                    # TensorBoard & tf.keras callbacks if necessary
                    callbacks = list(args[4])
                    callbacks, self.log_dir = _setup_callbacks(
                        callbacks, log_models, metrics_logger
                    )
                    # Replace the callbacks positional entry in the copied arguments and convert
                    # the arguments back to tuple form for usage in the training function
                    args[4] = callbacks
                    args = tuple(args)
                else:
                    # Make a shallow copy of the preexisting callbacks and introduce TensorBoard
                    # & tf.keras callbacks if necessary
                    callbacks = list(kwargs.get("callbacks") or [])
                    kwargs["callbacks"], self.log_dir = _setup_callbacks(
                        callbacks, log_models, metrics_logger
                    )

                result = original(inst, *args, **kwargs)

            _flush_queue()
            mlflow.log_artifacts(local_dir=self.log_dir.location, artifact_path="tensorboard_logs")
            if self.log_dir.is_temp:
                shutil.rmtree(self.log_dir.location)

            return result

        def _on_exception(self, exception):
            if (
                self.log_dir is not None
                and self.log_dir.is_temp
                and os.path.exists(self.log_dir.location)
            ):
                shutil.rmtree(self.log_dir.location)

    def add_event(original, self, event):
        _log_event(event)
        return original(self, event)

    def add_summary(original, self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _flush_queue()
        return result

    managed = [
        (tensorflow.estimator.Estimator, "train", train),
        (tensorflow.keras.Model, "fit", FitPatch),
    ]

    if Version(tensorflow.__version__) < Version("2.1.0"):
        # `fit_generator()` is deprecated in TF >= 2.1.0 and simply wraps `fit()`.
        # To avoid unintentional creation of nested MLflow runs caused by a patched
        # `fit_generator()` method calling a patched `fit()` method, we only patch
        # `fit_generator()` in TF < 2.1.0
        managed.append((tensorflow.keras.Model, "fit_generator", FitGeneratorPatch))

    non_managed = [
        (EventFileWriter, "add_event", add_event),
        (EventFileWriterV2, "add_event", add_event),
        (FileWriter, "add_summary", add_summary),
        (tensorflow.estimator.Estimator, "export_saved_model", export_saved_model),
        (tensorflow.estimator.Estimator, "export_savedmodel", export_saved_model),
    ]

    # Add compat.v1 Estimator patching for versions of tensfor that are 2.0+.
    if Version(tensorflow.__version__) >= Version("2.0.0"):
        old_estimator_class = tensorflow.compat.v1.estimator.Estimator
        v1_train = (old_estimator_class, "train", train)
        v1_export_saved_model = (old_estimator_class, "export_saved_model", export_saved_model)
        v1_export_savedmodel = (old_estimator_class, "export_savedmodel", export_saved_model)

        managed.append(v1_train)
        non_managed.append(v1_export_saved_model)
        non_managed.append(v1_export_savedmodel)

    for p in managed:
        safe_patch(FLAVOR_NAME, *p, manage_run=True)

    for p in non_managed:
        safe_patch(FLAVOR_NAME, *p)
