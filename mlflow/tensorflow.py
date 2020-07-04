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
import gorilla
import concurrent.futures
import warnings
import atexit
import time
import tempfile
from collections import namedtuple

import pandas

import mlflow
import tensorflow
import mlflow.keras
from distutils.version import LooseVersion
from contextlib import contextmanager
from tensorflow.keras.callbacks import Callback, TensorBoard  # pylint: disable=import-error
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import keyword_only, experimental
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params
from mlflow.entities import Metric


FLAVOR_NAME = "tensorflow"

_logger = logging.getLogger(__name__)

_MAX_METRIC_QUEUE_SIZE = 500

_LOG_EVERY_N_STEPS = 100

_metric_queue = []

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# For tracking if the run was started by autologging.
_AUTOLOG_RUN_ID = None


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(
        additional_conda_deps=[
            "tensorflow={}".format(tensorflow.__version__),
        ],
        additional_pip_deps=None,
        additional_conda_channels=None)


@keyword_only
def log_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, artifact_path,
              conda_env=None, signature: ModelSignature=None,
              input_example: ModelInputExample=None, registered_model_name=None):
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
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model. The
                      following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'tensorflow=1.8.0'
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

        :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.tensorflow,
                     tf_saved_model_dir=tf_saved_model_dir, tf_meta_graph_tags=tf_meta_graph_tags,
                     tf_signature_def_key=tf_signature_def_key, conda_env=conda_env,
                     registered_model_name=registered_model_name,
                     signature=signature,
                     input_example=input_example)


@keyword_only
def save_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, path,
               mlflow_model=None, conda_env=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None):
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
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model. The
                      following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'tensorflow=1.8.0'
                            ]
                        }
    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    """
    _logger.info(
        "Validating the specified TensorFlow model by attempting to load it in a new TensorFlow"
        " graph...")
    _validate_saved_model(tf_saved_model_dir=tf_saved_model_dir,
                          tf_meta_graph_tags=tf_meta_graph_tags,
                          tf_signature_def_key=tf_signature_def_key)
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
    shutil.move(os.path.join(path, root_relative_path), os.path.join(path, model_dir_subpath))

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(FLAVOR_NAME, saved_model_dir=model_dir_subpath,
                            meta_graph_tags=tf_meta_graph_tags,
                            signature_def_key=tf_signature_def_key)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _validate_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Validate the TensorFlow SavedModel by attempting to load it in a new TensorFlow graph.
    If the loading process fails, any exceptions thrown by TensorFlow are propagated.
    """
    if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
        validation_tf_graph = tensorflow.Graph()
        validation_tf_sess = tensorflow.Session(graph=validation_tf_graph)
        with validation_tf_graph.as_default():
            _load_tensorflow_saved_model(tf_saved_model_dir=tf_saved_model_dir,
                                         tf_sess=validation_tf_sess,
                                         tf_meta_graph_tags=tf_meta_graph_tags,
                                         tf_signature_def_key=tf_signature_def_key)
    else:
        _load_tensorflow_saved_model(tf_saved_model_dir=tf_saved_model_dir,
                                     tf_meta_graph_tags=tf_meta_graph_tags,
                                     tf_signature_def_key=tf_signature_def_key)


def load_model(model_uri, tf_sess=None):
    """
    Load an MLflow model that contains the TensorFlow flavor from the specified path.

    *With TensorFlow version <2.0.0, this method must be called within a TensorFlow graph context.*

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


    :param tf_sess: The TensorFlow session in which to load the model. If using TensorFlow
                    version >= 2.0.0, this argument is ignored. If using TensorFlow <2.0.0, if no
                    session is passed to this function, MLflow will attempt to load the model using
                    the default TensorFlow session.  If no default session is available, then the
                    function raises an exception.
    :return: For TensorFlow < 2.0.0, a TensorFlow signature definition of type:
             ``tensorflow.core.protobuf.meta_graph_pb2.SignatureDef``. This defines the input and
             output tensors for model inference.
             For TensorFlow >= 2.0.0, A callable graph (tf.function) that takes inputs and
             returns inferences.

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

    if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
        if not tf_sess:
            tf_sess = tensorflow.get_default_session()
            if not tf_sess:
                raise MlflowException("No TensorFlow session found while calling load_model()." +
                                      "You can set the default Tensorflow session before calling" +
                                      " load_model via `session.as_default()`, or directly pass " +
                                      "a session in which to load the model via the tf_sess " +
                                      "argument.")

    else:
        if tf_sess:
            warnings.warn("A TensorFlow session was passed into load_model, but the " +
                          "currently used version is TF 2.0 where sessions are deprecated. " +
                          "The tf_sess argument will be ignored.", FutureWarning)
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key =\
        _get_and_parse_flavor_configuration(model_path=local_model_path)
    return _load_tensorflow_saved_model(tf_saved_model_dir=tf_saved_model_dir,
                                        tf_meta_graph_tags=tf_meta_graph_tags,
                                        tf_signature_def_key=tf_signature_def_key,
                                        tf_sess=tf_sess)


def _load_tensorflow_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key,
                                 tf_sess=None):
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
    :param tf_sess: The TensorFlow session in which to load the metagraph.
                    Required in TensorFlow versions < 2.0.0. Unused in TensorFlow versions >= 2.0.0
    :return: For TensorFlow versions < 2.0.0:
             A TensorFlow signature definition of type:
             ``tensorflow.core.protobuf.meta_graph_pb2.SignatureDef``. This defines input and
             output tensors within the specified metagraph for inference.
             For TensorFlow versions >= 2.0.0:
             A callable graph (tensorflow.function) that takes inputs and returns inferences.
    """
    if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
        loaded = tensorflow.saved_model.loader.load(
            sess=tf_sess,
            tags=tf_meta_graph_tags,
            export_dir=tf_saved_model_dir)
        loaded_sig = loaded.signature_def
    else:
        loaded = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
                tags=tf_meta_graph_tags,
                export_dir=tf_saved_model_dir)
        loaded_sig = loaded.signatures
    if tf_signature_def_key not in loaded_sig:
        raise MlflowException("Could not find signature def key %s. Available keys are: %s"
                              % (tf_signature_def_key, list(loaded_sig.keys())))
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
    tf_saved_model_dir = os.path.join(model_path, flavor_conf['saved_model_dir'])
    tf_meta_graph_tags = flavor_conf['meta_graph_tags']
    tf_signature_def_key = flavor_conf['signature_def_key']
    return tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``. This function loads an MLflow
    model with the TensorFlow flavor into a new TensorFlow graph and exposes it behind the
    ``pyfunc.predict`` interface.

    :param path: Local filesystem path to the MLflow Model with the ``tensorflow`` flavor.
    """
    tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key =\
        _get_and_parse_flavor_configuration(model_path=path)
    if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
        tf_graph = tensorflow.Graph()
        tf_sess = tensorflow.Session(graph=tf_graph)
        with tf_graph.as_default():
            signature_def = _load_tensorflow_saved_model(
                tf_saved_model_dir=tf_saved_model_dir, tf_sess=tf_sess,
                tf_meta_graph_tags=tf_meta_graph_tags, tf_signature_def_key=tf_signature_def_key)

        return _TFWrapper(tf_sess=tf_sess, tf_graph=tf_graph, signature_def=signature_def)
    else:
        loaded_model = tensorflow.saved_model.load(  # pylint: disable=no-value-for-parameter
                                                   export_dir=tf_saved_model_dir,
                                                   tags=tf_meta_graph_tags)
        return _TF2Wrapper(infer=loaded_model.signatures[tf_signature_def_key])


class _TFWrapper(object):
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``. For TensorFlow versions < 2.0.0.
    """
    def __init__(self, tf_sess, tf_graph, signature_def):
        """
        :param tf_sess: The TensorFlow session used to evaluate the model.
        :param tf_graph: The TensorFlow graph containing the model.
        :param signature_def: The TensorFlow signature definition used to transform input dataframes
                              into tensors and output vectors into dataframes.
        """
        self.tf_sess = tf_sess
        self.tf_graph = tf_graph
        # We assume that input keys in the signature definition correspond to
        # input DataFrame column names
        self.input_tensor_mapping = {
            tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
            for tensor_column_name, tensor_info in signature_def.inputs.items()
        }
        # We assume that output keys in the signature definition correspond to
        # output DataFrame column names
        self.output_tensors = {
            sigdef_output: tf_graph.get_tensor_by_name(tnsr_info.name)
            for sigdef_output, tnsr_info in signature_def.outputs.items()
        }

    def predict(self, df):
        with self.tf_graph.as_default():
            # Build the feed dict, mapping input tensors to DataFrame column values.
            feed_dict = {
                self.input_tensor_mapping[tensor_column_name]: df[tensor_column_name].values
                for tensor_column_name in self.input_tensor_mapping.keys()
            }
            raw_preds = self.tf_sess.run(self.output_tensors, feed_dict=feed_dict)
            pred_dict = {column_name: values.ravel() for
                         column_name, values in raw_preds.items()}
            return pandas.DataFrame(data=pred_dict)


class _TF2Wrapper(object):
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``. For TensorFlow versions >= 2.0.0.
    """
    def __init__(self, infer):
        """
        :param infer: Tensorflow function returned by a saved model that is used for inference.
        """
        self.infer = infer

    def predict(self, df):
        feed_dict = {}
        for df_col_name in list(df):
            # If there are multiple columns with the same name, selecting the shared name
            # from the DataFrame will result in another DataFrame containing the columns
            # with the shared name. TensorFlow cannot make eager tensors out of pandas
            # DataFrames, so we convert the DataFrame to a numpy array here.
            val = df[df_col_name]
            if isinstance(val, pandas.DataFrame):
                val = val.values
            feed_dict[df_col_name] = tensorflow.constant(val)
        raw_preds = self.infer(**feed_dict)
        pred_dict = {
            col_name: raw_preds[col_name].numpy() for col_name in raw_preds.keys()
        }
        for col in pred_dict.keys():
            if all(len(element) == 1 for element in pred_dict[col]):
                pred_dict[col] = pred_dict[col].ravel()
            else:
                pred_dict[col] = pred_dict[col].tolist()

        return pandas.DataFrame.from_dict(data=pred_dict)


class __MLflowTfKerasCallback(Callback):
    """
        Callback for auto-logging parameters (we rely on TensorBoard for metrics) in TensorFlow < 2.
        Records model structural information as params after training finishes.
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        opt = self.model.optimizer
        if hasattr(opt, '_name'):
            try_mlflow_log(mlflow.log_param, 'optimizer_name', opt._name)
        # Elif checks are if the optimizer is a TensorFlow optimizer rather than a Keras one.
        elif hasattr(opt, 'optimizer'):
            # TensorFlow optimizer parameters are associated with the inner optimizer variable.
            # Therefore, we assign opt to be opt.optimizer for logging parameters.
            opt = opt.optimizer
            try_mlflow_log(mlflow.log_param, 'optimizer_name', type(opt).__name__)
        if hasattr(opt, 'lr'):
            lr = opt.lr if type(opt.lr) is float else tensorflow.keras.backend.eval(opt.lr)
            try_mlflow_log(mlflow.log_param, 'learning_rate', lr)
        elif hasattr(opt, '_lr'):
            lr = opt._lr if type(opt._lr) is float else tensorflow.keras.backend.eval(opt._lr)
            try_mlflow_log(mlflow.log_param, 'learning_rate', lr)
        if hasattr(opt, 'epsilon'):
            epsilon = opt.epsilon if type(opt.epsilon) is float \
                else tensorflow.keras.backend.eval(opt.epsilon)
            try_mlflow_log(mlflow.log_param, 'epsilon', epsilon)
        elif hasattr(opt, '_epsilon'):
            epsilon = opt._epsilon if type(opt._epsilon) is float \
                else tensorflow.keras.backend.eval(opt._epsilon)
            try_mlflow_log(mlflow.log_param, 'epsilon', epsilon)

        sum_list = []
        self.model.summary(print_fn=sum_list.append)
        summary = '\n'.join(sum_list)
        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, "model_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary)
            try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):  # pylint: disable=unused-argument
        try_mlflow_log(mlflow.keras.log_model, self.model, artifact_path='model')


class __MLflowTfKeras2Callback(Callback):
    """
        Callback for auto-logging parameters and metrics in TensorFlow >= 2.0.0.
        Records model structural information as params when training starts.
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        config = self.model.optimizer.get_config()
        for attribute in config:
            try_mlflow_log(mlflow.log_param, "opt_" + attribute, config[attribute])

        sum_list = []
        self.model.summary(print_fn=sum_list.append)
        summary = '\n'.join(sum_list)
        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, "model_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary)
            try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch-1) % _LOG_EVERY_N_STEPS == 0:
            try_mlflow_log(mlflow.log_metrics, logs, step=epoch)

    def on_train_end(self, logs=None):  # pylint: disable=unused-argument
        try_mlflow_log(mlflow.keras.log_model, self.model, artifact_path='model')


def _log_artifacts_with_warning(**kwargs):
    try_mlflow_log(mlflow.log_artifacts, **kwargs)


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
    global _metric_queue
    client = mlflow.tracking.MlflowClient()
    dic = _assoc_list_to_map(_metric_queue)
    for key in dic:
        try_mlflow_log(client.log_batch, key, metrics=dic[key], params=[], tags=[])
    _metric_queue = []


atexit.register(_flush_queue)


def _add_to_queue(key, value, step, time, run_id):
    """
    Add a metric to the metric queue. Flush the queue if it exceeds
    max size.
    """
    met = Metric(key=key, value=value, timestamp=time, step=step)
    _metric_queue.append((run_id, met))
    if len(_metric_queue) > _MAX_METRIC_QUEUE_SIZE:
        _flush_queue()


def _log_event(event):
    """
    Extracts metric information from the event protobuf
    """
    if not mlflow.active_run():
        try_mlflow_log(mlflow.start_run)
        global _AUTOLOG_RUN_ID
        _AUTOLOG_RUN_ID = mlflow.active_run().info.run_id
    if event.WhichOneof('what') == 'summary':
        summary = event.summary
        for v in summary.value:
            if v.HasField('simple_value'):
                if (event.step-1) % _LOG_EVERY_N_STEPS == 0:
                    _thread_pool.submit(_add_to_queue, key=v.tag,
                                        value=v.simple_value, step=event.step,
                                        time=int(time.time() * 1000),
                                        run_id=mlflow.active_run().info.run_id)


def _get_tensorboard_callback(lst):
    for x in lst:
        if isinstance(x, tensorflow.keras.callbacks.TensorBoard):
            return x
    return None


# A representation of a TensorBoard event logging directory with two attributes:
# :location - string: The filesystem location of the logging directory
# :is_temp - boolean: `True` if the logging directory was created for temporary use by MLflow,
#                     `False` otherwise
_TensorBoardLogDir = namedtuple("_TensorBoardLogDir", ["location", "is_temp"])


def _setup_callbacks(lst):
    """
    Adds TensorBoard and MlfLowTfKeras callbacks to the
    input list, and returns the new list and appropriate log directory.
    """
    tb = _get_tensorboard_callback(lst)
    if tb is None:
        log_dir = _TensorBoardLogDir(location=tempfile.mkdtemp(), is_temp=True)
        out_list = lst + [TensorBoard(log_dir.location)]
    else:
        log_dir = _TensorBoardLogDir(location=tb.log_dir, is_temp=False)
        out_list = lst
    if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
        out_list += [__MLflowTfKerasCallback()]
    else:
        out_list += [__MLflowTfKeras2Callback()]
    return out_list, log_dir


@experimental
def autolog(every_n_iter=100):
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

    :param every_n_iter: The frequency with which metrics should be logged.
                                  Defaults to 100. Ex: a value of 100 will log metrics
                                  at step 0, 100, 200, etc.
    """
    global _LOG_EVERY_N_STEPS
    _LOG_EVERY_N_STEPS = every_n_iter

    if LooseVersion(tensorflow.__version__) < LooseVersion('1.12'):
        warnings.warn("Could not log to MLflow. Only TensorFlow versions" +
                      "1.12 <= v <= 2.0.0 are supported.")
        return

    try:
        from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
        from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.summary.writer.writer import FileWriter
    except ImportError:
        warnings.warn("Could not log to MLflow. Only TensorFlow versions" +
                      "1.12 <= v <= 2.0.0 are supported.")
        return

    @contextmanager
    def _manage_active_run():
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            global _AUTOLOG_RUN_ID
            if mlflow.active_run() is not None:  # defensive check in case `mlflow.start_run` fails
                _AUTOLOG_RUN_ID = mlflow.active_run().info.run_id
        yield mlflow.active_run()
        if mlflow.active_run() is not None and mlflow.active_run().info.run_id == _AUTOLOG_RUN_ID:
            try_mlflow_log(mlflow.end_run)

    @gorilla.patch(tensorflow.estimator.Estimator)
    def train(self, *args, **kwargs):
        with _manage_active_run():
            original = gorilla.get_original_attribute(tensorflow.estimator.Estimator, 'train')

            # Checking step and max_step parameters for logging
            if len(args) >= 3:
                try_mlflow_log(mlflow.log_param, 'steps', args[2])
                if len(args) >= 4:
                    try_mlflow_log(mlflow.log_param, 'max_steps', args[3])
            if 'steps' in kwargs:
                try_mlflow_log(mlflow.log_param, 'steps', kwargs['steps'])
            if 'max_steps' in kwargs:
                try_mlflow_log(mlflow.log_param, 'max_steps', kwargs['max_steps'])

            result = original(self, *args, **kwargs)

            return result

    @gorilla.patch(tensorflow.estimator.Estimator)
    def export_saved_model(self, *args, **kwargs):
        auto_end = False
        if not mlflow.active_run():
            global _AUTOLOG_RUN_ID
            if _AUTOLOG_RUN_ID:
                try_mlflow_log(mlflow.start_run, _AUTOLOG_RUN_ID)
            else:
                try_mlflow_log(mlflow.start_run)
                auto_end = True

        original = gorilla.get_original_attribute(tensorflow.estimator.Estimator,
                                                  'export_saved_model')
        serialized = original(self, *args, **kwargs)
        try_mlflow_log(log_model, tf_saved_model_dir=serialized.decode('utf-8'),
                       tf_meta_graph_tags=[tag_constants.SERVING],
                       tf_signature_def_key='predict',
                       artifact_path='model')
        if (mlflow.active_run() is not None and mlflow.active_run().info.run_id == _AUTOLOG_RUN_ID)\
                or auto_end:
            try_mlflow_log(mlflow.end_run)
        return serialized

    @gorilla.patch(tensorflow.estimator.Estimator)
    def export_savedmodel(self, *args, **kwargs):
        auto_end = False
        global _AUTOLOG_RUN_ID
        if not mlflow.active_run():
            if _AUTOLOG_RUN_ID:
                try_mlflow_log(mlflow.start_run, _AUTOLOG_RUN_ID)
            else:
                try_mlflow_log(mlflow.start_run)
                auto_end = True

        original = gorilla.get_original_attribute(tensorflow.estimator.Estimator,
                                                  'export_savedmodel')
        serialized = original(self, *args, **kwargs)
        try_mlflow_log(log_model, tf_saved_model_dir=serialized.decode('utf-8'),
                       tf_meta_graph_tags=[tag_constants.SERVING],
                       tf_signature_def_key='predict',
                       artifact_path='model')
        if (mlflow.active_run() is not None and mlflow.active_run().info.run_id == _AUTOLOG_RUN_ID)\
                or auto_end:
            try_mlflow_log(mlflow.end_run)
        return serialized

    def _early_stop_check(callbacks):
        for callback in callbacks:
            if isinstance(callback, tensorflow.keras.callbacks.EarlyStopping):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'monitor': callback.monitor,
                                        'min_delta': callback.min_delta,
                                        'patience': callback.patience,
                                        'baseline': callback.baseline,
                                        'restore_best_weights': callback.restore_best_weights}
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            try_mlflow_log(mlflow.log_metric, 'stopped_epoch', stopped_epoch)
            # Weights are restored only if early stopping occurs
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                try_mlflow_log(mlflow.log_metric, 'restored_epoch', restored_epoch)
                restored_metrics = {key: history.history[key][restored_epoch]
                                    for key in history.history.keys()}
                # Metrics are logged as 'epoch_loss' and 'epoch_acc' in TF 1.X
                if LooseVersion(tensorflow.__version__) < LooseVersion('2.0.0'):
                    if 'loss' in restored_metrics:
                        restored_metrics['epoch_loss'] = restored_metrics.pop('loss')
                    if 'acc' in restored_metrics:
                        restored_metrics['epoch_acc'] = restored_metrics.pop('acc')
                # Checking that a metric history exists
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    try_mlflow_log(mlflow.log_metrics, restored_metrics, step=last_epoch)

    @gorilla.patch(tensorflow.keras.Model)
    def fit(self, *args, **kwargs):
        with _manage_active_run():
            original = gorilla.get_original_attribute(tensorflow.keras.Model, 'fit')

            unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']

            log_fn_args_as_params(original, args, kwargs, unlogged_params)
            early_stop_callback = None

            # Checking if the 'callback' argument of fit() is set
            if len(args) >= 6:
                tmp_list = list(args)
                early_stop_callback = _early_stop_check(tmp_list[5])
                tmp_list[5], log_dir = _setup_callbacks(tmp_list[5])
                args = tuple(tmp_list)
            elif 'callbacks' in kwargs:
                early_stop_callback = _early_stop_check(kwargs['callbacks'])
                kwargs['callbacks'], log_dir = _setup_callbacks(kwargs['callbacks'])
            else:
                kwargs['callbacks'], log_dir = _setup_callbacks([])

            _log_early_stop_callback_params(early_stop_callback)

            history = original(self, *args, **kwargs)

            _log_early_stop_callback_metrics(early_stop_callback, history)

            _flush_queue()
            _log_artifacts_with_warning(
                local_dir=log_dir.location, artifact_path='tensorboard_logs')
            if log_dir.is_temp:
                shutil.rmtree(log_dir.location)

            return history

    @gorilla.patch(tensorflow.keras.Model)
    def fit_generator(self, *args, **kwargs):
        with _manage_active_run():
            original = gorilla.get_original_attribute(tensorflow.keras.Model, 'fit_generator')

            unlogged_params = ['self', 'generator', 'callbacks', 'validation_data', 'verbose']

            log_fn_args_as_params(original, args, kwargs, unlogged_params)

            # Checking if the 'callback' argument of fit() is set
            if len(args) >= 5:
                tmp_list = list(args)
                tmp_list[4], log_dir = _setup_callbacks(tmp_list[4])
                args = tuple(tmp_list)
            elif 'callbacks' in kwargs:
                kwargs['callbacks'], log_dir = _setup_callbacks(kwargs['callbacks'])
            else:
                kwargs['callbacks'], log_dir = _setup_callbacks([])
            result = original(self, *args, **kwargs)
            _flush_queue()
            _log_artifacts_with_warning(
                local_dir=log_dir.location, artifact_path='tensorboard_logs')
            if log_dir.is_temp:
                shutil.rmtree(log_dir.location)

            return result

    @gorilla.patch(EventFileWriter)
    def add_event(self, event):
        _log_event(event)
        original = gorilla.get_original_attribute(EventFileWriter, 'add_event')
        return original(self, event)

    @gorilla.patch(FileWriter)
    def add_summary(self, *args, **kwargs):
        original = gorilla.get_original_attribute(FileWriter, 'add_summary')
        result = original(self, *args, **kwargs)
        _flush_queue()
        return result

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patches = [
        gorilla.Patch(EventFileWriter, 'add_event', add_event, settings=settings),
        gorilla.Patch(EventFileWriterV2, 'add_event', add_event, settings=settings),
        gorilla.Patch(tensorflow.estimator.Estimator, 'train', train, settings=settings),
        gorilla.Patch(tensorflow.keras.Model, 'fit', fit, settings=settings),
        gorilla.Patch(tensorflow.keras.Model, 'fit_generator', fit_generator, settings=settings),
        gorilla.Patch(tensorflow.estimator.Estimator, 'export_saved_model',
                      export_saved_model, settings=settings),
        gorilla.Patch(tensorflow.estimator.Estimator, 'export_savedmodel',
                      export_savedmodel, settings=settings),
        gorilla.Patch(FileWriter, 'add_summary', add_summary, settings=settings),
        ]

    for x in patches:
        gorilla.apply(x)
