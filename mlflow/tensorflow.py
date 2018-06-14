"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

import os

import pandas
import tensorflow as tf

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


class _TFWrapper(object):
    """
    Wrapper class that creates a predict function such that 
    predict(data: pandas.DataFrame) -> pandas.DataFrame
    """
    def __init__(self, saved_model_dir, signature_def_name=None):
        self._saved_model_dir = saved_model_dir
        self._signature_def_name = signature_def_name

    def predict(self, df):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, 
                                                        [tf.saved_model.tag_constants.SERVING], 
                                                        self._saved_model_dir)
            if not self._signature_def_name:
                """
                TODO: add support for replacing "predict" with 
                #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                """ 
                self._signature_def_name = "predict"
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, 
                                                                      self._signature_def_name)

            # Determining input tensors.
            feed_mapping = {}
            feed_names = []
            for sigdef_key, tnsr_info in sig_def.inputs.items():
                tnsr_name = tnsr_info.name
                feed_mapping[sigdef_key] = tnsr_name
                feed_names.append(tnsr_name)

            # Determining output tensors.
            fetch_mapping = {}
            fetch_names = []
            for sigdef_key, tnsr_info in sig_def.outputs.items():
                tnsr_name = tnsr_info.name
                fetch_mapping[sigdef_key] = tnsr_name
                fetch_names.append(tnsr_name)

            fetches = [graph.get_tensor_by_name(t_name) for t_name in fetch_names]

            feed_dict = {}
            for col in feed_mapping:
                tnsr_name = feed_mapping[col]
                feed_dict[graph.get_tensor_by_name(tnsr_name)] = df[col].values
            data = sess.run(fetches, feed_dict=feed_dict)
            return pandas.DataFrame(data=data[0])


def log_saved_model(saved_model_dir, artifact_path):
    """Log a Tensorflow model as an MLflow artifact for the current run."""
    run_id = mlflow.tracking.active_run().info.run_uuid
    mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow")
    mlflow_model.add_flavor("tensorflow")
    mlflow_model.save(os.path.join(saved_model_dir, "MLmodel"))
    mlflow.tracking.log_artifacts(saved_model_dir, artifact_path)


def load_pyfunc(saved_model_dir, signature_def_name=None):
    return _TFWrapper(saved_model_dir, signature_def_name)
