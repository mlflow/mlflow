from __future__ import print_function

import os
import dill as pickle
import pandas
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

from mlflow import tensorflow, pyfunc
from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir

def load_pyfunc(path):
    model_fn = None
    model_dir = None
    for filename in os.listdir(path):
        if filename == "model_fn.pkl":
            with open(os.path.join(path, filename), "rb") as f:
                model_fn = pickle.load(f)
                print("found function")
        elif filename == "model_dir.pkl":
            with open(os.path.join(path, filename), "rb") as f:
                model_dir = pickle.load(f)
                print("found dir")
    return tf.estimator.Estimator(model_fn, model_dir=model_dir)


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._trainingFeatures = {}
        self._feature_names = iris.feature_names[:2]
        for i in range(0, 2):
            tab = str.maketrans(dict.fromkeys(' ()'))
            iris.feature_names[i] = iris.feature_names[i].translate(tab)
            self._trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
        tf_feat_cols = []
        self._feature_names = iris.feature_names[:2]
        for col in iris.feature_names[:2]:
            tf_feat_cols.append(tf.feature_column.numeric_column(col))
        self._input_train = tf.estimator.inputs.numpy_input_fn(self._trainingFeatures, 
                                                                    self._y, 
                                                                    shuffle=False, 
                                                                    batch_size=1)
        self._dnn = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols, 
                                                hidden_units=[1])
        self._dnn.train(self._input_train, steps=100)
        self._dnn_predict = self._dnn.predict(self._input_train)

    def test_model_save_load(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            path = tmp.path("dnn")
            tensorflow.save_model(tf_model=self._dnn, path=path)
            x = tensorflow.load_model(path)
            xpred = x.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
            saved = []
            for s in self._dnn_predict:
                saved.append(s['predictions'])
            loaded = []
            for index, rows in xpred.iterrows():
                loaded.append(rows)
            np.testing.assert_array_equal(saved, loaded)
            # sklearn should also be stored as a valid pyfunc model
            # test pyfunc compatibility
            y = pyfunc.load_pyfunc(path)
            ypred = y.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
            loaded = []
            for index, rows in ypred.iterrows():
                loaded.append(rows)
            np.testing.assert_array_equal(saved, loaded)

    def test_model_log(self):
        with TempDir(chdr=True, remove_on_exit=True):
            tracking.start_run()
            try:
                tensorflow.log_model(tf_model=self._dnn, artifact_path="dnn")
                x = tensorflow.load_model("dnn", run_id=tracking.active_run().info.run_uuid)
                xpred = x.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
                saved = []
                for s in self._dnn_predict:
                    saved.append(s['predictions'])
                loaded = []
                for index, rows in xpred.iterrows():
                    loaded.append(rows)
                np.testing.assert_array_equal(saved, loaded)
            finally:
                tracking.end_run()


if __name__ == '__main__':
    unittest.main()
