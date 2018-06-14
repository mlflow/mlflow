from __future__ import print_function

import os
import dill as pickle
import pandas
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

from mlflow import tensorflow, pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


class TestModelExport(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._trainingFeatures = {}
        self._feature_names = iris.feature_names[:2]
        for i in range(0, 2):
            # Tensorflow is fickle about feature names, so we remove offending characters
            iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
            iris.feature_names[i] = iris.feature_names[i].replace("(", "")
            iris.feature_names[i] = iris.feature_names[i].replace(")", "")
            self._trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
        self._tf_feat_cols = []
        self._feature_names = iris.feature_names[:2]
        # Creating Tensorflow-specific numeric columns for input.
        for col in iris.feature_names[:2]:
            self._tf_feat_cols.append(tf.feature_column.numeric_column(col))
        # Creating input training function.
        self._input_train = tf.estimator.inputs.numpy_input_fn(self._trainingFeatures, 
                                                                    self._y, 
                                                                    shuffle=False, 
                                                                    batch_size=1)
        # Creating Deep Neural Network Regressor. 
        self._dnn = tf.estimator.DNNRegressor(feature_columns=self._tf_feat_cols, 
                                                hidden_units=[1])
        self._sess = tf.Session()
        # Training and creating expected predictions on training dataset.
        self._dnn.train(self._input_train, steps=100)
        self._dnn_predict = self._dnn.predict(self._input_train)

    def test_log_saved_model(self):
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            tracking.start_run()
            try:
                # Creating dict of features names (str) to placeholders (tensors)
                feature_spec = {}
                for name in self._feature_names:
                    feature_spec[name] = tf.placeholder("float", name=name, shape=[150])
                # Creating receiver function for model saving.
                receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                saved_model_path = tmp.path("model")
                os.makedirs(saved_model_path)
                os.makedirs(tmp.path("hello"))
                # Saving Tensorflow model.
                saved_model_path = self._dnn.export_savedmodel(saved_model_path, 
                                                               receiver_fn).decode("utf-8")
                # Logging the Tensorflow model just saved.
                tensorflow.log_saved_model(saved_model_dir=saved_model_path, 
                                           artifact_path=tmp.path("hello"))
                # Loading the saved Tensorflow model as a pyfunc.
                x = tensorflow.load_pyfunc(saved_model_path, "predict")
                # Predicting on the iris dataset using the pyfunc.
                xpred = x.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
                saved = []
                for s in self._dnn_predict:
                    saved.append(s['predictions'])
                loaded = []
                for index, rows in xpred.iterrows():
                    loaded.append(rows)
                # Asserting that the loaded model predictions are as expected.
                np.testing.assert_array_equal(saved, loaded)
            finally:
                tracking.end_run()
