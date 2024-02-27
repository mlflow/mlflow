import argparse
import pickle

import numpy as np
import tensorflow as tf

import mlflow.tensorflow
from mlflow.utils.file_utils import TempDir

parser = argparse.ArgumentParser()

parser.add_argument("--tracking_uri")
parser.add_argument("--task_type")
parser.add_argument("--save_as_type")
parser.add_argument("--save_path")

args = parser.parse_args()
mlflow.set_tracking_uri(args.tracking_uri)

tf.random.set_seed(1337)

inputs = tf.keras.layers.Input(shape=3, name="features", dtype=tf.float64)
outputs = tf.keras.layers.Dense(2)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=[outputs])

task_type = args.task_type
save_as_type = args.save_as_type

run_id = None

if save_as_type == "tf1-estimator":
    with TempDir() as tmp:
        tf.saved_model.save(model, tmp.path())
        if task_type == "save_model":
            save_path = args.save_path

            mlflow.tensorflow.save_model(
                tf_saved_model_dir=tmp.path(),
                tf_meta_graph_tags=["serve"],
                tf_signature_def_key="serving_default",
                path=save_path,
                extra_pip_requirements=["protobuf<4.0.0"],
            )
        elif task_type == "log_model":
            with mlflow.start_run() as run:
                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=tmp.path(),
                    tf_meta_graph_tags=["serve"],
                    tf_signature_def_key="serving_default",
                    artifact_path="model",
                    extra_pip_requirements=["protobuf<4.0.0"],
                )
                run_id = run.info.run_id
        else:
            raise ValueError("Illegal arguments.")
elif save_as_type == "keras":
    if task_type == "save_model":
        save_path = args.save_path
        mlflow.keras.save_model(model, save_path)
    elif task_type == "log_model":
        with mlflow.start_run() as run:
            mlflow.keras.log_model(model, "model")
            run_id = run.info.run_id
    else:
        raise ValueError("Illegal arguments.")


inference_data = np.array([[2.0, 3.0, 4.0], [11.0, 12.0, 13.0]], dtype=np.float64)
expected_results_data = model.predict(inference_data)

output_data_info = (inference_data, expected_results_data, run_id)
output_data_file_path = "output_data.pkl"

with open(output_data_file_path, "wb") as f:
    pickle.dump(output_data_info, f)
