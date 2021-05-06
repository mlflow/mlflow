import os

import numpy as np
import tensorflow as tf

import mlflow


def test_tf_saved_model_model(tmpdir):
    tf.random.set_seed(1337)

    mlflow_model_path = os.path.join(str(tmpdir), "mlflow_model")
    tf_model_path = os.path.join(str(tmpdir), "tf_model")

    # Build TensorFlow model.
    inputs = tf.keras.layers.Input(shape=1, name="feature1", dtype=tf.float32)
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs])

    # Save model in TensorFlow SavedModel format.
    tf.saved_model.save(model, tf_model_path)

    # Save TensorFlow SavedModel as MLflow model.
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=tf_model_path,
        tf_meta_graph_tags=["serve"],
        tf_signature_def_key="serving_default",
        path=mlflow_model_path,
    )

    def load_and_predict():
        model_uri = mlflow_model_path
        mlflow_model = mlflow.pyfunc.load_model(model_uri)
        feed_dict = {"feature1": tf.constant([[2.0]])}
        predictions = mlflow_model.predict(feed_dict)
        assert np.allclose(predictions["dense"], np.asarray([-0.09599352]))

    load_and_predict()
