"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

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
            print(tf.saved_model.tag_constants.SERVING)
            print(sess)
            print(self._saved_model_dir)
            return df
