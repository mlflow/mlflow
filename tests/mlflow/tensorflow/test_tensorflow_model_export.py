from __future__ import print_function

import collections
import os
import pandas
import unittest

import pandas as pd
import sklearn.datasets as datasets
import tensorflow as tf

from mlflow import tensorflow, pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


class TestModelExport(unittest.TestCase):

    def helper(self, feature_spec, tmp, estimator, df):
        """
        This functions handles exporting, logging, loading back, and predicting on an estimator for 
        testing purposes.
        """
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        saved_estimator_path = tmp.path("model")
        os.makedirs(saved_estimator_path)
        # Saving TensorFlow model.
        saved_estimator_path = estimator.export_savedmodel(saved_estimator_path, 
                                                       receiver_fn).decode("utf-8")
        # Logging the TensorFlow model just saved.
        tensorflow.log_saved_model(saved_model_dir=saved_estimator_path,
                                   signature_def_key="predict", 
                                   artifact_path=tmp.path("hello"))
        # Loading the saved TensorFlow model as a pyfunc.
        x = pyfunc.load_pyfunc(saved_estimator_path)
        # Predicting on the dataset using the pyfunc.
        return x.predict(df)


    def test_log_saved_model(self):
        # This tests model logging capabilities on the sklearn.iris dataset.
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            iris = datasets.load_iris()
            X = iris.data[:, :2]  # we only take the first two features.
            y = iris.target
            trainingFeatures = {}
            for i in range(0, 2):
                # TensorFlow is fickle about feature names, so we remove offending characters
                iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
                iris.feature_names[i] = iris.feature_names[i].replace("(", "")
                iris.feature_names[i] = iris.feature_names[i].replace(")", "")
                trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
            tf_feat_cols = []
            feature_names = iris.feature_names[:2]
            # Creating TensorFlow-specific numeric columns for input.
            for col in iris.feature_names[:2]:
                tf_feat_cols.append(tf.feature_column.numeric_column(col))
            # Creating input training function.
            input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures,
                                                             y,
                                                             shuffle=False,
                                                             batch_size=1)
            # Creating Deep Neural Network Regressor.
            estimator = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols,
                                                  hidden_units=[1])
            # Training and creating expected predictions on training dataset.
            estimator.train(input_train, steps=10)
            # Saving the estimator's prediction on the training data; assume the DNNRegressor
            # produces a single output column named 'predictions'
            pred_col = "predictions"
            estimator_preds = [s[pred_col] for s in estimator.predict(input_train)]
            estimator_preds_df = pd.DataFrame({pred_col: estimator_preds})
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
                pyfunc_preds_df = self.helper(feature_spec, tmp, estimator,
                                              pandas.DataFrame(data=X, columns=feature_names))

                # Asserting that the loaded model predictions are as expected.
                assert estimator_preds_df.equals(pyfunc_preds_df)
            finally:
                # Restoring the old logging location.
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_dir)


    def test_categorical_columns(self):
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
                ("body-style", [""]),
                ("curb-weight", [0.0]),
                ("highway-mpg", [0.0]),
                ("price", [0.0])
            ])

            types = collections.OrderedDict((key, type(value[0]))
                                            for key, value in defaults.items())
            df = pandas.read_csv(path, names=types.keys(), dtype=types, na_values="?")
            df = df.dropna()

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
            estimator = tf.estimator.DNNRegressor(
                hidden_units=[20, 20], feature_columns=feature_columns)

            # Training the estimator.
            estimator.train(input_fn=input_train, steps=10)
            # Saving the estimator's prediction on the training data; assume the DNNRegressor
            # produces a single output column named 'predictions'
            pred_col = "predictions"
            estimator_preds = [s[pred_col] for s in estimator.predict(input_train)]
            estimator_preds_df = pd.DataFrame({pred_col: estimator_preds})
            # Setting the logging such that it is in the temp folder and deleted after the test.
            old_tracking_dir = tracking.get_tracking_uri()
            tracking_dir = os.path.abspath(tmp.path("mlruns"))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            tracking.start_run()
            try:
                # Creating dict of features names (str) to placeholders (tensors)
                feature_spec = {}
                feature_spec["body-style"] = tf.placeholder("string", 
                                                            name="body-style", 
                                                            shape=[None])
                feature_spec["curb-weight"] = tf.placeholder("float", 
                                                            name="curb-weight", 
                                                            shape=[None])
                feature_spec["highway-mpg"] = tf.placeholder("float", 
                                                            name="highway-mpg", 
                                                            shape=[None])

                pyfunc_preds_df = self.helper(feature_spec, tmp, estimator, df)
                # Asserting that the loaded model predictions are as expected. Allow for some
                # imprecision as this is expected with TensorFlow.
                pandas.testing.assert_frame_equal(
                    pyfunc_preds_df, estimator_preds_df, check_less_precise=6)
            finally:
                # Restoring the old logging location.
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_dir)
