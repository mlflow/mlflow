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

# Command-line arguments
parser = argparse.ArgumentParser(description='Tensorflow Example')
# parser.add_argument('--hidden_units', nargs='*', default='50 20'.split( ), metavar='L',
#                    help='hidden units (default: [50, 20])')  #  https://docs.python.org/3/library/argparse.html#nargs
parser.add_argument('--hidden_units', type=str, default='50 20', metavar='L',
                    help='hidden units (default: "50 20")')
parser.add_argument('--steps', type=int, default=1000, metavar='N',
                    help='steps (default: 1000)')
parser.add_argument('--tag_constants', type=str, default=tf.saved_model.tag_constants.SERVING, metavar='S',
                    help='tf.saved_model.tag_constants (default: tf.saved_model.tag_constants.SERVING)')
args = parser.parse_args()

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    #    tf.app.run(main=main)

    with mlflow.start_run() as tracked_run:
        tf.logging.set_verbosity(tf.logging.INFO)
        # for key, value in vars(args).items():
        #    mlflow.log_param(key, value)
        hidden_units = [ int( s ) for s in args.hidden_units.split( ) ]  #  [50, 20]
        steps = args.steps  #  1000
        print( '(hdnNts, stp) = ' , hidden_units , steps )

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
            tensorflow.log_model( tf_saved_model_dir = saved_estimator_path , tf_signature_def_key = "predict" ,
                tf_meta_graph_tags = [ args.tag_constants ] , artifact_path="model")
            # Reloading the model
            pyfunc_model = pyfunc.load_pyfunc( os.path.join(mlflow.get_artifact_uri() , 'model' ) )  #  (saved_estimator_path)
            df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
            # Predicting on the loaded Python Function
            predict_df = pyfunc_model.predict(df)
            predict_df['original_labels'] = y_test
            print(predict_df)
        finally:
            shutil.rmtree(temp)
