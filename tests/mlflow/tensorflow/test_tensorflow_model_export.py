from __future__ import print_function

import collections
import os
import pandas
import unittest

import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets
import shutil

from mlflow import tensorflow, pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


class TestModelExport(unittest.TestCase):

    def helper(self, feature_spec, tmp, model, model_predict, df):
        # This functions handles 
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        saved_model_path = tmp.path("model")
        os.makedirs(saved_model_path)
        os.makedirs(tmp.path("hello"))
        # Saving Tensorflow model.
        saved_model_path = model.export_savedmodel(saved_model_path, 
                                                       receiver_fn).decode("utf-8")
        # Logging the Tensorflow model just saved.
        tensorflow.log_saved_model(saved_model_dir=saved_model_path,
                                   signature_def_key="predict", 
                                   artifact_path=tmp.path("hello"))
        # Loading the saved Tensorflow model as a pyfunc.
        x = pyfunc.load_pyfunc(saved_model_path)
        # Predicting on the dataset using the pyfunc.
        xpred = x.predict(df)
        saved = []
        for s in model_predict:
            saved.append(s['predictions'])
        loaded = []
        for index, rows in xpred.iterrows():
            loaded.append(rows)
        return (saved, loaded)

    def test_log_saved_model(self):
        # This tests model logging capabilities on the sklearn.iris dataset.
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            iris = datasets.load_iris()
            X = iris.data[:, :2]  # we only take the first two features.
            y = iris.target
            trainingFeatures = {}
            feature_names = iris.feature_names[:2]
            for i in range(0, 2):
                # Tensorflow is fickle about feature names, so we remove offending characters
                iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
                iris.feature_names[i] = iris.feature_names[i].replace("(", "")
                iris.feature_names[i] = iris.feature_names[i].replace(")", "")
                trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
            tf_feat_cols = []
            feature_names = iris.feature_names[:2]
            # Creating Tensorflow-specific numeric columns for input.
            for col in iris.feature_names[:2]:
                tf_feat_cols.append(tf.feature_column.numeric_column(col))
            # Creating input training function.
            input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures, 
                                                                        y, 
                                                                        shuffle=False, 
                                                                        batch_size=1)
            # Creating Deep Neural Network Regressor. 
            dnn = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols, 
                                                    hidden_units=[1])
            sess = tf.Session()
            # Training and creating expected predictions on training dataset.
            dnn.train(input_train, steps=100)
            dnn_predict = dnn.predict(input_train)
            # Setting the logging such that it is in the temp folder and deleted after the test.
            old_tracking_dir = tracking.get_tracking_uri()
            tracking_dir = os.path.abspath(tmp.path("mlruns"))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            tracking.start_run()
            try:
                # Creating dict of features names (str) to placeholders (tensors)
                feature_spec = {}
                for name in feature_names:
                    feature_spec[name] = tf.placeholder("float", name=name, shape=[150])

                results = self.helper(feature_spec, tmp, dnn, dnn_predict, pandas.DataFrame(data=X, columns=feature_names))

                # Asserting that the loaded model predictions are as expected.
                np.testing.assert_array_equal(results[0], results[1])
            finally:
                # Restoring the old logging location.
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_dir)


    def test_cat_cols(self):
        """
        This tests logging capabilities on datasets with categorical columns.
        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/get_started/regression/imports85.py
        for reference code.
        """
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            # Downloading the data into a pandas DataFrame.
            URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
            path = tf.contrib.keras.utils.get_file(URL.split("/")[-1], URL)
            # Order is important for the csv-readers, so we use an OrderedDict here.
            defaults = collections.OrderedDict([
                ("symboling", [0]),
                ("normalized-losses", [0.0]),
                ("make", [""]),
                ("fuel-type", [""]),
                ("aspiration", [""]),
                ("num-of-doors", [""]),
                ("body-style", [""]),
                ("drive-wheels", [""]),
                ("engine-location", [""]),
                ("wheel-base", [0.0]),
                ("length", [0.0]),
                ("width", [0.0]),
                ("height", [0.0]),
                ("curb-weight", [0.0]),
                ("engine-type", [""]),
                ("num-of-cylinders", [""]),
                ("engine-size", [0.0]),
                ("fuel-system", [""]),
                ("bore", [0.0]),
                ("stroke", [0.0]),
                ("compression-ratio", [0.0]),
                ("horsepower", [0.0]),
                ("peak-rpm", [0.0]),
                ("city-mpg", [0.0]),
                ("highway-mpg", [0.0]),
                ("price", [0.0])
            ])

            types = collections.OrderedDict((key, type(value[0]))
                                            for key, value in defaults.items())
            df = pandas.read_csv(path, names=types.keys(), dtype=types, na_values="?")
            df = df.dropna()
            # Keeping only the data we are testing.
            df = df[["price", "body-style", "curb-weight", "highway-mpg"]]

            # Extract the label from the features dataframe.
            y_train = df.pop("price")

            # Creating the input training function required.
            trainingFeatures = {}

            for i in df:
                trainingFeatures[i] = df[i].values

            input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures, 
                                                            y_train.values, 
                                                            shuffle=False, 
                                                            batch_size=1)

            # Creating the feature columns required for the DNNRegressor.
            body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
            body_style = tf.feature_column.categorical_column_with_vocabulary_list(
                key="body-style", vocabulary_list=body_style_vocab)
            feature_columns = [
            tf.feature_column.numeric_column(key="curb-weight"),
            tf.feature_column.numeric_column(key="highway-mpg"),
            # Since this is a DNN model, convert categorical columns from sparse
            # to dense.
            # Wrap them in an `indicator_column` to create a
            # one-hot vector from the input.
            tf.feature_column.indicator_column(body_style),]

            # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
            # defined above as input.
            model = tf.estimator.DNNRegressor(
                hidden_units=[20, 20], feature_columns=feature_columns)

            # Training the model.
            model.train(input_fn=input_train, steps=100)
            # Saving the model's prediction on the training data.
            model_predict = model.predict(input_train)
            # Setting the logging such that it is in the temp folder and deleted after the test.
            old_tracking_dir = tracking.get_tracking_uri()
            tracking_dir = os.path.abspath(tmp.path("mlruns"))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            tracking.start_run()
            try:
                # Creating dict of features names (str) to placeholders (tensors)
                feature_spec = {}
                size = len(y_train.values)
                feature_spec["body-style"] = tf.placeholder("string", 
                                                            name="body-style", 
                                                            shape=[size])
                feature_spec["curb-weight"] = tf.placeholder("float", 
                                                            name="curb-weight", 
                                                            shape=[size])
                feature_spec["highway-mpg"] = tf.placeholder("float", 
                                                            name="highway-mpg", 
                                                            shape=[size])

                results = self.helper(feature_spec, tmp, model, model_predict, df)

                # Asserting that the loaded model predictions are as expected.
                # Tensorflow is known to have precision errors, hence the almost_equal.
                np.testing.assert_array_almost_equal(results[0], results[1], decimal = 2)
            finally:
                # Restoring the old logging location.
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_dir)
