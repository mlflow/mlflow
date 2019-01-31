"""
The ``mlflow.tensorflow`` module provides an API for logging and loading TensorFlow models.
This module exports TensorFlow models with the following flavors:

TensorFlow (native) format
    This is the main flavor that can be loaded back into TensorFlow.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import shutil
import yaml
import logging

import pandas
import tensorflow as tf

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "tensorflow"

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "tensorflow={}".format(tf.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)


_logger = logging.getLogger(__name__)


def log_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, artifact_path,
              conda_env=None):
    """
    Log a *serialized* collection of TensorFlow graphs and variables as an MLflow model
    for the current run. This method operates on TensorFlow variables and graphs that have been
    serialized in TensorFlow's ``SavedModel`` format. For more information about ``SavedModel``
    format, see the TensorFlow documentation:
    https://www.tensorflow.org/guide/saved_model#save_and_restore_models.

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
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.tensorflow.DEFAULT_CONDA_ENV``. If ``None``, the default
                      ``mlflow.tensorflow.DEFAULT_CONDA_ENV`` environment will be added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'tensorflow=1.8.0'
                            ]
                        }

    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.tensorflow,
                     tf_saved_model_dir=tf_saved_model_dir, tf_meta_graph_tags=tf_meta_graph_tags,
                     tf_signature_def_key=tf_signature_def_key, conda_env=conda_env)


def save_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key, path,
               mlflow_model=Model(), conda_env=None):
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
    :param mlflow_model: MLflow model configuration to which this flavor will be added.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.tensorflow.DEFAULT_CONDA_ENV``. If ``None``, the default
                      ``mlflow.tensorflow.DEFAULT_CONDA_ENV`` environment will be added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'tensorflow=1.8.0'
                            ]
                        }

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
    root_relative_path = _copy_file_or_tree(src=tf_saved_model_dir, dst=path, dst_dir=None)
    model_dir_subpath = "tfmodel"
    shutil.move(os.path.join(path, root_relative_path), os.path.join(path, model_dir_subpath))

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(FLAVOR_NAME, saved_model_dir=model_dir_subpath,
                            meta_graph_tags=tf_meta_graph_tags,
                            signature_def_key=tf_signature_def_key)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _validate_saved_model(tf_saved_model_dir, tf_meta_graph_tags, tf_signature_def_key):
    """
    Validate the TensorFlow SavedModel by attempting to load it in a new TensorFlow graph.
    If the loading process fails, any exceptions thrown by TensorFlow will be propagated.
    """
    validation_tf_graph = tf.Graph()
    validation_tf_sess = tf.Session(graph=validation_tf_graph)
    with validation_tf_graph.as_default():
        _load_model(tf_saved_model_dir=tf_saved_model_dir,
                    tf_sess=validation_tf_sess,
                    tf_meta_graph_tags=tf_meta_graph_tags,
                    tf_signature_def_key=tf_signature_def_key)


def load_model(path, tf_sess, run_id=None):
    """
    Load an MLflow model that contains the TensorFlow flavor from the specified path.

    **This method must be called within a TensorFlow graph context.**

    :param path: The local filesystem path or run-relative artifact path to the model.
    :param tf_sess: The TensorFlow session in which to the load the model.
    :return: A TensorFlow signature definition of type:
             ``tensorflow.core.protobuf.meta_graph_pb2.SignatureDef``. This defines the input and
             output tensors for model inference.

    >>> import mlflow.tensorflow
    >>> import tensorflow as tf
    >>> tf_graph = tf.Graph()
    >>> tf_sess = tf.Session(graph=tf_graph)
    >>> with tf_graph.as_default():
    >>>     signature_definition = mlflow.tensorflow.load_model(path="model_path", tf_sess=tf_sess)
    >>>     input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
    >>>                      for _, input_signature in signature_def.inputs.items()]
    >>>     output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
    >>>                       for _, output_signature in signature_def.outputs.items()]
    """
    if run_id is not None:
        path = _get_model_log_dir(model_name=path, run_id=run_id)
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    tf_saved_model_dir = os.path.join(path, flavor_conf['saved_model_dir'])
    return _load_model(tf_saved_model_dir=tf_saved_model_dir, tf_sess=tf_sess,
                       tf_meta_graph_tags=flavor_conf['meta_graph_tags'],
                       tf_signature_def_key=flavor_conf['signature_def_key'])


def _load_model(tf_saved_model_dir, tf_sess, tf_meta_graph_tags, tf_signature_def_key):
    """
    Load a specified TensorFlow model consisting of a TensorFlow meta graph and signature definition
    from a serialized TensorFlow ``SavedModel`` collection.

    :param tf_saved_model_dir: The local filesystem path or run-relative artifact path to the model.
    :param tf_sess: The TensorFlow session in which to the load the metagraph.
    :param tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized `SavedModel` object. For more information, see the `tags`
                               parameter of the `tf.saved_model.builder.SavedModelBuilder` method:
                               https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/
                               SavedModelBuilder#add_meta_graph
    :param tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized `SavedModel`'s signature
                                 definition mapping. For more information, see the
                                 `signature_def_map` parameter of the
                                 `tf.saved_model.builder.SavedModelBuilder` method.
    :return: A TensorFlow signature definition of type:
             ``tensorflow.core.protobuf.meta_graph_pb2.SignatureDef``. This defines input and
             output tensors within the specified metagraph for inference.
    """
    meta_graph_def = tf.saved_model.loader.load(
            sess=tf_sess,
            tags=tf_meta_graph_tags,
            export_dir=tf_saved_model_dir)
    if tf_signature_def_key not in meta_graph_def.signature_def:
        raise MlflowException("Could not find signature def key %s" % tf_signature_def_key)
    return meta_graph_def.signature_def[tf_signature_def_key]


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``. This function loads an MLflow
    model with the TensorFlow flavor into a new TensorFlow graph and exposes it behind the
    `pyfunc.predict` interface.
    """
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = load_model(path=path, tf_sess=tf_sess, run_id=None)

    return _TFWrapper(tf_sess=tf_sess, tf_graph=tf_graph, signature_def=signature_def)


class _TFWrapper(object):
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    predict(data: pandas.DataFrame) -> pandas.DataFrame.
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
        # We assume that input keys in the signature definition correspond to input DataFrame column
        # names
        self.input_tensor_mapping = {
                tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
                for tensor_column_name, tensor_info in signature_def.inputs.items()
        }
        # We assume that output keys in the signature definition correspond to output DataFrame
        # column names
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
            pred_dict = {column_name: values.ravel() for column_name, values in raw_preds.items()}
            return pandas.DataFrame(data=pred_dict)
