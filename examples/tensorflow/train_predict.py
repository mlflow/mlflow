from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import mlflow
from mlflow import tensorflow, tracking, pyfunc
import numpy as np
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
import os.path

parser = argparse.ArgumentParser(description='Tensorflow Example')
parser.add_argument('--hidden-units', type=str, default='50 20',
                    help='The number of hidden units in each of the two hidden unit layers (default: %(default)s)')
parser.add_argument('--steps', type=int, default=1000,
                    help='The number of training steps (default: %(default)d)')
args = parser.parse_args()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    with mlflow.start_run() as tracked_run:
        tf.logging.set_verbosity(tf.logging.INFO)
        hidden_units = [int(s) for s in args.hidden_units.split()]
        print("Hidden Units: {}".format(hidden_units))
        print("Training Steps: {}".format(args.steps))
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("training_steps", args.steps)

        # Builds, trains and evaluates a tf.estimator. Then, exports it for inference, logs the exported model
        # with MLflow, and loads the fitted model back as a PyFunc to make predictions.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
        # There are 13 features we are using for inference.
        feat_cols = [tf.feature_column.numeric_column(key="features", shape=(x_train.shape[1],))]
        feat_spec = {"features": tf.placeholder("float", name="features", shape=[None, x_train.shape[1]])}
        regressor = tf.estimator.DNNRegressor(hidden_units = hidden_units , feature_columns=feat_cols)
        train_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_train}, y_train, num_epochs=None, shuffle=True)

        regressor.train(train_input_fn, steps=args.steps)
        test_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_test}, y_test, num_epochs=None, shuffle=True)
        # Compute mean squared error
        mse = regressor.evaluate(test_input_fn, steps=args.steps)
        mlflow.log_metric("Mean Square Error", mse['average_loss'])
        # Building a receiver function for exporting
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
        temp = tempfile.mkdtemp()
        try:
            saved_estimator_path = regressor.export_savedmodel(temp, receiver_fn).decode("utf-8")
            # Logging the saved model
            tensorflow.log_model(tf_saved_model_dir=saved_estimator_path, tf_signature_def_key="predict", artifact_path="model", \
                tf_meta_graph_tags=[tf.saved_model.tag_constants.SERVING])  #  adding required parameter tf_meta_graph_tags
            # Reloading the model
            pyfunc_model = pyfunc.load_pyfunc(os.path.join(mlflow.get_artifact_uri(), 'model'))
            df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
            # Predicting on the loaded Python Function
            predict_df = pyfunc_model.predict(df)
            predict_df['original_labels'] = y_test
            print(predict_df)
        finally:
            shutil.rmtree(temp)
