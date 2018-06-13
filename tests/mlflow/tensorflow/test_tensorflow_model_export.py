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
            # Tensorflow is fickle about feature names, so we remove offending characters
            iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
            iris.feature_names[i] = iris.feature_names[i].replace("(", "")
            iris.feature_names[i] = iris.feature_names[i].replace(")", "")
            self._trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
        self._tf_feat_cols = []
        self._feature_names = iris.feature_names[:2]
        for col in iris.feature_names[:2]:
            self._tf_feat_cols.append(tf.feature_column.numeric_column(col))
        self._input_train = tf.estimator.inputs.numpy_input_fn(self._trainingFeatures, 
                                                                    self._y, 
                                                                    shuffle=False, 
                                                                    batch_size=1)
        self._dnn = tf.estimator.DNNRegressor(feature_columns=self._tf_feat_cols, 
                                                hidden_units=[1])
        self._sess = tf.Session()
        self._dnn.train(self._input_train, steps=100)
        self._dnn_predict = self._dnn.predict(self._input_train)

    def test_log_saved_model(self):
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            tracking.start_run()
            try:
                # feature_spec = tf.feature_column.make_parse_example_spec(self._tf_feat_cols)
                # print("FEATURE_SPEC:", feature_spec)
                feature_spec = {}
                for name in self._feature_names:
                    feature_spec[name] = tf.placeholder("float", name=name, shape=[150])
                receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                saved_model_path = tmp.path("model")
                os.makedirs(saved_model_path)
                os.makedirs(tmp.path("hello"))
                saved_model_path = self._dnn.export_savedmodel(saved_model_path, receiver_fn).decode("utf-8")
                # os.makedirs(tmp.path("hello"))
                # saved_model_path = tmp.path("model")
                # x = tf.placeholder("float", name="sepallengthcm")
                # y = tf.placeholder("float", name="sepalwidthcm")
                # z = tf.placeholder("float", name="z")
                # #print("check model/ has files", os.listdir(saved_model_path))
                # tf.saved_model.simple_save(self._sess, saved_model_path,
                #                            inputs={"sepallengthcm":x, "sepalwidthcm":y},
                #                            outputs={"z":z})
                tensorflow.log_saved_model(saved_model_dir=saved_model_path, artifact_path=tmp.path("hello"))
                x = tensorflow.load_pyfunc(saved_model_path, "predict")
                # print("data:", self._X)
                # print("columns:", self._feature_names)
                # print("X:", x)
                print("DataFrame:", pandas.DataFrame(data=self._X, columns=self._feature_names))
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

    # def test_model_save_load(self):
    #     with TempDir(chdr=True, remove_on_exit=True) as tmp:
    #         path = tmp.path("dnn")
    #         tensorflow.save_model(tf_model=self._dnn, path=path)
    #         x = tensorflow.load_model(path)
    #         xpred = x.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
    #         saved = []
    #         for s in self._dnn_predict:
    #             saved.append(s['predictions'])
    #         loaded = []
    #         for index, rows in xpred.iterrows():
    #             loaded.append(rows)
    #         np.testing.assert_array_equal(saved, loaded)
    #         # sklearn should also be stored as a valid pyfunc model
    #         # test pyfunc compatibility
    #         y = pyfunc.load_pyfunc(path)
    #         ypred = y.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
    #         loaded = []
    #         for index, rows in ypred.iterrows():
    #             loaded.append(rows)
    #         np.testing.assert_array_equal(saved, loaded)

    # def test_model_log(self):
    #     with TempDir(chdr=True, remove_on_exit=True):
    #         tracking.start_run()
    #         try:
    #             tensorflow.log_model(tf_model=self._dnn, artifact_path="dnn")
    #             x = tensorflow.load_model("dnn", run_id=tracking.active_run().info.run_uuid)
    #             xpred = x.predict(pandas.DataFrame(data=self._X, columns=self._feature_names))
    #             saved = []
    #             for s in self._dnn_predict:
    #                 saved.append(s['predictions'])
    #             loaded = []
    #             for index, rows in xpred.iterrows():
    #                 loaded.append(rows)
    #             np.testing.assert_array_equal(saved, loaded)
    #         finally:
    #             tracking.end_run()


if __name__ == '__main__':
    unittest.main()
