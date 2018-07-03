from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlflow import tensorflow, tracking
import numpy as np
import pandas as pd
import shutil
import tempfile
import tensorflow as tf


def main(argv):
    # Builds, trains and evaluates a tf.estimator. Then, exports it for inference, logs the exported model 
    # with MLflow, and loads the fitted model back as a PyFunc to make predictions.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    feat_cols = [tf.feature_column.numeric_column(key="features", shape=(13,))]
    feat_spec = {"features":tf.placeholder("float", name="features", shape=[102, 13])}
    regressor = tf.estimator.DNNRegressor(
    hidden_units=[50, 20],
    feature_columns=feat_cols)
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_train}, y_train, batch_size=404, num_epochs=None, shuffle=True)
    regressor.train(train_input_fn, steps=1000)
    test_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_test}, y_test, batch_size=404, num_epochs=None, shuffle=True)
    # Compute mean squared error
    regressor.evaluate(test_input_fn, steps=100)
    # Building a receiver function for exporting
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
    temp = tempfile.mkdtemp()
    try:
        tracking.start_run()
        saved_estimator_path = regressor.export_savedmodel(temp, receiver_fn).decode("utf-8")
        # Logging the saved model
        tensorflow.log_saved_model(saved_model_dir=saved_estimator_path, signature_def_key="predict", artifact_path="model")
        # Reloading the model
        pyfunc = tensorflow.load_pyfunc(saved_estimator_path)
        df = pd.DataFrame(data=x_test, columns=["features"] * 13)
        # Predicting on the loaded Python Function
        predict_df = pyfunc.predict(df)
        print(predict_df)
    finally:
        shutil.rmtree(temp)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)