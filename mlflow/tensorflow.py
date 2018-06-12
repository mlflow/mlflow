"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

import os
import dill as pickle

import pandas
import tensorflow as tf

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


class TFWrapper(object):
    """Wrapper class that creates a predict function such that 
    predict(data: pandas.DataFrame) -> pandas.DataFrame"""
    def __init__(self, sess, fetches):
        self._sess = sess
        self._fetches = fetches

    def predict(self, df):
        data = self._sess.run(self.fetches, feed_dict=pandas.DataFrame.to_dict(df))
        return pandas.DataFrame.from_dict(data)


def log_saved_model(saved_model_dir, artifact_path):
    """Log a Tensorflow model as an MLflow artifact for the current run."""
    with TempDir() as tmp:
        local_path = tmp.path("model")
        run_id = mlflow.tracking.active_run().info.run_uuid
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow")
        mlflow_model.add_flavor("tensorflow")
        mlflow_model.save(os.path.join(path, "MLmodel"))
        mlflow.tracking.log_artifacts(local_path, artifact_path)


def load_pyfunc(saved_model_dir, signature_def_name=None):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, 
                                                    [tf.saved_model.tag_constants.SERVING], 
                                                    saved_model_dir)
        if not signature_def_name:
            signature_def_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, signature_def_name)

        feed_mapping = {}
        feed_names = []
        for sigdef_key, tnsr_info in sig_def.inputs.items():
            tnsr_name = tnsr_info.name
            feed_mapping[sigdef_key] = tnsr_name
            feed_names.append(tnsr_name)

        fetch_mapping = {}
        fetch_names = []
        for sigdef_key, tnsr_info in sig_def.outputs.items():
            tnsr_name = tnsr_info.name
            fetch_mapping[sigdef_key] = tnsr_name
            fetch_names.append(tnsr_name)

        fetches = [tf.get_tensor_by_name(tnsr_name) for tnsr_name in fetch_names]
        return TFWrapper(sess, fetches)

