# in case this is run outside of conda environment with python2
import mlflow
from mlflow import pyfunc
import pandas as pd
import argparse
import shutil
import tempfile
import tensorflow as tf
import mlflow.tensorflow

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps", default=1000, type=int, help="number of steps used for training and evaluation"
)


def main(argv):
    with mlflow.start_run():
        args = parser.parse_args(argv[1:])

        # Builds, trains and evaluates a tf.estimator. Then, exports it for inference,
        # logs the exported model with MLflow, and loads the fitted model back as a PyFunc.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

        # There are 13 features we are using for inference.
        feat_cols = [tf.feature_column.numeric_column(key="features", shape=(x_train.shape[1],))]
        feat_spec = {
            "features": tf.placeholder("float", name="features", shape=[None, x_train.shape[1]])
        }

        hidden_units = [50, 20]
        steps = args.steps

        regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feat_cols)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {"features": x_train}, y_train, num_epochs=None, shuffle=True
        )
        regressor.train(train_input_fn, steps=steps)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {"features": x_test}, y_test, num_epochs=None, shuffle=True
        )
        # Compute mean squared error
        mse = regressor.evaluate(test_input_fn, steps=steps)

        # Building a receiver function for exporting
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
        temp = tempfile.mkdtemp()
        try:
            # The model is automatically logged when export_saved_model() is called.
            saved_estimator_path = regressor.export_savedmodel(temp, receiver_fn).decode("utf-8")

            # Since the model was automatically logged as an artifact (more specifically
            # a MLflow Model), we don't need to use saved_estimator_path to load back the model.
            # MLflow takes care of it!
            pyfunc_model = pyfunc.load_model(mlflow.get_artifact_uri("model"))
            df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])

            # Checking the PyFunc's predictions are the same as the original model's predictions.
            predict_df = pyfunc_model.predict(df)
            predict_df["original_labels"] = y_test
            print(predict_df)
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
