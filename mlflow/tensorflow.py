"""
The ``mlflow.tensorflow`` module provides an API for logging and loading TensorFlow models
as :py:mod:`mlflow.pyfunc` models.

You must save your own ``saved_model`` and pass its
path to ``log_saved_model(saved_model_dir)``. To load the model to predict on it, you call
``model = pyfunc.load_pyfunc(saved_model_dir)`` followed by
``prediction = model.predict(pandas DataFrame)`` to obtain a prediction in a pandas DataFrame.

The loaded :py:mod:`mlflow.pyfunc` model *does not* expose any APIs for model training.
"""

from __future__ import absolute_import

import os
import shutil

import pandas
import tensorflow as tf

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import _copy_file_or_tree

FLAVOR_NAME = "tensorflow"


def log_model(tf_saved_model_dir, meta_graph_tags, signature_def_key, artifact_path, 
              conda_env=None):
    return Model.log(artifact_path=artifact_path, flavor=mlflow.tensorflow,
                     tf_saved_model_dir=tf_saved_model_dir, meta_graph_tags=meta_graph_tags,
                     signature_def_key=signature_def_key, conda_env=conda_env)


def save_model(tf_saved_model_dir, meta_graph_tags, signature_def_key, path, mlflow_model=Model(), 
               conda_env=None):
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path), DIRECTORY_NOT_EMPTY)
    os.makedirs(path)
    root_relative_path = _copy_file_or_tree(src=tf_saved_model_dir, dst=path, dst_dir=None)
    model_dir_subpath = "tfmodel"
    shutil.move(os.path.join(path, root_relative_path), os.path.join(path, model_dir_subpath))

    mlflow_model.add_flavor(FLAVOR_NAME, saved_model_dir=model_dir_subpath,
                            meta_graph_tags=meta_graph_tags, signature_def_key=signature_def_key)

    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        _copy_file_or_tree(src=conda_env, dst=os.path.join(path, model_conda_env))

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", env=model_conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def load_model(path, tf_sess, tf_graph, tf_context=None, run_id=None):
    if run_id is not None:
        path = _get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(os.path.join(path, 'MLmodel'))
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    saved_model_dir = os.path.join(path, conf['saved_model_dir'])
    return _load_model(saved_model_dir=saved_model_dir, tf_sess=tf_sess, tf_graph=tf_graph,
                       tf_context=tf_context, meta_graph_tags=conf['meta_graph_tags'], 
                       signature_def_key=conf['signature_def_key'])


def _load_model(saved_model_dir, tf_sess, tf_graph, meta_graph_tags, signature_def_key, 
                tf_context=None):
    if tf_context is None:
        tf_context = tf_graph.as_default()
    with tf_context:
        meta_graph_def = tf.saved_model.loader.load(
                sess=tf_sess, 
                tags=meta_graph_tags, 
                export_dir=saved_model_dir)
        signature_def = tf.contrib.saved_model.get_signature_def_by_key(
                meta_graph_def, signature_def_key)
        return signature_def


def _load_pyfunc(path, tf_sess=None, tf_graph=None, tf_context=None):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    if tf_graph is None:
        tf_graph = tf.get_default_graph()
    if tf_sess is None:
        tf_sess = tf.Session(graph=tf_graph)
    if tf_context is None:
        tf_context = tf_graph.as_default()

    signature_def = load_model(path=path, tf_sess=tf_sess, tf_graph=tf_graph, 
                               tf_context=tf_context, run_id=None)

    return _TFWrapper(tf_sess=tf_sess, tf_graph=tf_graph, 
                      tf_context=tf_context, signature_def=signature_def)


class _TFWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pandas.DataFrame) -> pandas.DataFrame
    """
    def __init__(self, tf_sess, tf_graph, tf_context, signature_def):
        self.tf_sess = tf_sess
        self.tf_context = tf_context
        # We assume that input keys in the signature definition correspond to input DataFrame column
        # names
        self.input_tensor_mapping = {
                tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
                for tensor_column_name, tensor_info in signature_def.inputs().items()
        }
        # We assume that output keys in the signature definition correspond to output DataFrame
        # column names
        self.output_tensors = {
                sigdef_output: tf_graph.get_tensor_by_name(tnsr_info.name)
                for sigdef_output, tnsr_info in signature_def.outputs.items()
        }

    def predict(self, df):
        with self.tf_context:
            # Build the feed dict, mapping input tensors to DataFrame column values.
            feed_dict = {
                    self.input_tensor_mapping[tensor_column_name]: df[tensor_column_name].values
                    for tensor_column_name in self.input_tensor_mapping.keys()
            }
            raw_preds = self.sess.run(self.output_tensors, feed_dict=feed_dict)
            pred_dict = {column_name: values.ravel() for column_name, values in raw_preds.items()}
            return pandas.DataFrame(data=pred_dict)
