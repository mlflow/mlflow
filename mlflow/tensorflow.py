"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

import pandas
import tensorflow as tf


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
                # TODO: add support for replacing "predict" with 
                # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                self._signature_def_name = "predict"
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, 
                                                                      self._signature_def_name)
