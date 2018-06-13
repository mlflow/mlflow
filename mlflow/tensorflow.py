"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

import os
import dill as pickle

import numpy as np
import pandas
import tensorflow as tf

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


class TFWrapper(object):
    """Wrapper class that creates a predict function such that 
    predict(data: pandas.DataFrame) -> pandas.DataFrame"""
    def __init__(self, saved_model_dir, signature_def_name):
        self._saved_model_dir = saved_model_dir
        self._signature_def_name = signature_def_name

    def predict(self, df):
        self._graph = tf.Graph()
        with tf.Session(graph=self._graph) as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, 
                                                        [tf.saved_model.tag_constants.SERVING], 
                                                        self._saved_model_dir)
            if not self._signature_def_name:
                self._signature_def_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, self._signature_def_name)
            for key in meta_graph_def.signature_def:
                print("STORED SIG DEF KEY:", key)
                print("STORED SIG DEF KEY contents:", meta_graph_def.signature_def[key])
            print("CHOSEN SIG DEF KEY:", self._signature_def_name)
            self._feed_mapping = {}
            feed_names = []
            for sigdef_key, tnsr_info in sig_def.inputs.items():
                print("SIF DEF INPUT ITEM:", sigdef_key, tnsr_info)
                tnsr_name = tnsr_info.name
                self._feed_mapping[sigdef_key] = tnsr_name
                feed_names.append(tnsr_name)

            print("FEED_MAPPING:", self._feed_mapping)    
            print("FEED_NAMES:", feed_names)

            fetch_mapping = {}
            fetch_names = []
            for sigdef_key, tnsr_info in sig_def.outputs.items():
                tnsr_name = tnsr_info.name
                fetch_mapping[sigdef_key] = tnsr_name
                fetch_names.append(tnsr_name)

            print("FETCH_MAPPING:", fetch_mapping)
            print("FETCH_NAMES:", fetch_names)

            self._fetches = [self._graph.get_tensor_by_name(tnsr_name) for tnsr_name in fetch_names]

            feed_dict = {}
            for col in self._feed_mapping:
                # placeholder = tf.placeholder("float", shape=None, name=col)
                feed_dict[self._graph.get_tensor_by_name(self._feed_mapping[col])] = df[col].values#np.expand_dims(df[col].values,
                                                                                     #               axis=-1)
            print("fetches:", self._fetches)
            print("feed_dict:", feed_dict)
            # init = tf.global_variables_initializer()
            # sess.run(init)
            data = sess.run(self._fetches, feed_dict=feed_dict)
            print("DATA:", data)
            return pandas.DataFrame(data=data[0])


def log_saved_model(saved_model_dir, artifact_path):
    """Log a Tensorflow model as an MLflow artifact for the current run."""
    with TempDir() as tmp:
        run_id = mlflow.tracking.active_run().info.run_uuid
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow")
        mlflow_model.add_flavor("tensorflow")
        mlflow_model.save(os.path.join(saved_model_dir, "MLmodel"))
        mlflow.tracking.log_artifacts(saved_model_dir, artifact_path)


def load_pyfunc(saved_model_dir, signature_def_name=None):
    return TFWrapper(saved_model_dir, signature_def_name)

