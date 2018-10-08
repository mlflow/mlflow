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

import pandas
import tensorflow as tf

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking.fluent import _get_or_start_run, log_artifacts
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import _copy_file_or_tree

FLAVOR_NAME = "tensorflow"


def log_model(tf_saved_model_dir, artifact_path, signature_def_key=None, conda_env=None):
    return Model.log(artifact_path=artifact_path, flavor=mlflow.tensorflow,
                     tf_saved_model_dir=tf_saved_model_dir, signature_def_key=signature_def_key,
                     conda_env=conda_env)


def save_model(tf_saved_model_dir, path, signature_def_key=None, mlflow_model=Model(),
               conda_env=None):
    model_dir_subpath = "model"
    _copy_file_or_tree(src=tf_saved_model_dir, dst=path, dst_dir=model_dir_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, saved_model_dir=model_dir_subpath,
                            signature_def_key=signature_def_key)

    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        _copy_file_or_tree(src=conda_env, dst=os.path.join(path, model_conda_env))

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow", env=model_conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def load_model(path, sess, graph, run_id=None):
    if run_id is not None:
        path = _get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(os.path.join(path, 'MLmodel'))
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    saved_model_dir = os.path.join(path, conf['saved_model_dir'])
    signature_def_key = conf.get('signature_def_key', None)
    return _load_model(saved_model_dir=saved_model_dir, sess=sess, graph=graph,
                       signature_def_key=signature_def_key)


def _load_model(saved_model_dir, sess, graph, signature_def_key=None):
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                saved_model_dir)
    signature_def = tf.contrib.saved_model.get_signature_def_by_key(
            meta_graph_def,
            signature_def_key if signature_def_key is not None else
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    return signature_def


def _load_pyfunc(path, sess=None, graph=None):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    if graph is None:
        graph = tf.Graph()
    if sess is None:
        sess = tf.Session(graph=graph)

    with graph:
        signature_def = load_model(path=path, sess=sess, graph=graph)

    return _TFWrapper(sess=sess, graph=graph, signature_def=signature_def)


class _TFWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pandas.DataFrame) -> pandas.DataFrame


    """
    def __init__(self, sess, graph, signature_def):
        self.sess = sess
        self.graph = graph
        self.input_tensor_mapping = {
                tensor_column_name: graph.get_tensor_by_name(tensor_info.name)
                for tensor_column_name, tensor_info in signature_def.inputs().items()
        }
        self.output_tensors = {
                sigdef_output: graph.get_tensor_by_name(tnsr_info.name)
                for sigdef_output, tnsr_info in signature_def.outputs.items()
        }

    def predict(self, df):
        with self.graph:
            feed_dict = {
                    self.input_tensor_mapping[tensor_column_name]: df[tensor_column_name].values
                    for tensor_column_name in self.input_tensor_mapping.keys()
            }
            raw_preds = self.sess.run(self.output_tensors, feed_dict=feed_dict)
            pred_dict = {column_name: list(values) for column_name, values in raw_preds.items()}
            return pandas.DataFrame(data=pred_dict)
