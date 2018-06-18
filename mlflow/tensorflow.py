"""MLflow integration for Tensorflow."""

from __future__ import absolute_import

import tensorflow as tf


def predict(df):
    graph = tf.Graph()
    return graph, df
