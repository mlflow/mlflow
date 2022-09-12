# pep8: disable=E501

import collections
import os
from pathlib import Path
import pickle
import pytest
import json

import numpy as np
import pandas as pd
import pandas.testing
import tensorflow as tf
import iris_data_utils

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.tensorflow import _TF2Wrapper
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.pyfunc.backend import _execute_in_conda_env

from tests.helper_functions import pyfunc_serve_and_score_model


@pytest.fixture
def tf_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "pytest"])
    return conda_env


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


ModelDataInfo = collections.namedtuple(
    "ModelDataInfo",
    [
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
        "run_id",
    ]
)


def save_or_log_tf_model_by_mlflow128(model_type, task_type, model_path=None):
    conda_env = get_or_create_conda_env("tests/tensorflow/mlflow-128-tf-23-env.yaml")
    with TempDir() as tmpdir:
        output_data_file_path = tmpdir.path("output_data.pkl")
        # change cwd to avoid it imports current repo mlflow.
        _execute_in_conda_env(
            conda_env,
            f"python tests/tensorflow/save_tf_estimator_model.py {model_type} {task_type} "
            f"{output_data_file_path} {model_path if model_path else ''}",
            install_mlflow=False,
            command_env={"MLFLOW_TRACKING_URI": mlflow.tracking.get_tracking_uri()}
        )
        with open(output_data_file_path, "rb") as f:
            return ModelDataInfo(*pickle.load(f))


def test_load_model_from_remote_uri_succeeds(model_path, mock_s3_bucket):
    model_data_info = save_or_log_tf_model_by_mlflow128("iris", "save_model", model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    infer = mlflow.tensorflow.load_model(model_uri=model_uri)
    feed_dict = {
        df_column_name: tf.constant(model_data_info.inference_df[df_column_name])
        for df_column_name in list(model_data_info.inference_df)
    }
    raw_preds = infer(**feed_dict)
    pred_dict = {column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()}
    for col in pred_dict:
        np.testing.assert_allclose(
            np.array(pred_dict[col], dtype=np.float),
            np.array(model_data_info.raw_results[col], dtype=np.float),
        )


def test_iris_model_can_be_loaded_and_evaluated_successfully(model_path):
    model_data_info = save_or_log_tf_model_by_mlflow128("iris", "save_model", model_path)

    def load_and_evaluate():

        infer = mlflow.tensorflow.load_model(model_uri=model_path)
        feed_dict = {
            df_column_name: tf.constant(model_data_info.inference_df[df_column_name])
            for df_column_name in list(model_data_info.inference_df)
        }
        raw_preds = infer(**feed_dict)
        pred_dict = {
            column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()
        }
        for col in pred_dict:
            np.testing.assert_array_equal(pred_dict[col], model_data_info.raw_results[col])

    load_and_evaluate()

    with tf.device("/CPU:0"):
        load_and_evaluate()


def test_log_and_load_model_persists_and_restores_model_successfully():
    model_data_info = save_or_log_tf_model_by_mlflow128("iris", "log_model")
    model_uri = f"runs:/{model_data_info.run_id}/model"
    mlflow.tensorflow.load_model(model_uri=model_uri)


def test_iris_data_model_can_be_loaded_and_evaluated_as_pyfunc(model_path):
    model_data_info = save_or_log_tf_model_by_mlflow128("iris", "save_model", model_path)

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(model_data_info.inference_df)
    assert isinstance(results_df, pd.DataFrame)
    for key in results_df.keys():
        np.testing.assert_array_equal(results_df[key], model_data_info.raw_df[key])

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(model_data_info.inference_df):
        inp_dict[df_col_name] = model_data_info.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    for key in results.keys():
        np.testing.assert_array_equal(results[key], model_data_info.raw_df[key].tolist())

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(model_data_info.inference_df):
        inp_list.append(model_data_info.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_categorical_model_can_be_loaded_and_evaluated_as_pyfunc(model_path):
    model_data_info = save_or_log_tf_model_by_mlflow128("categorical", "save_model", model_path)

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(model_data_info.inference_df)
    # Precision is less accurate for the categorical model when we load back the saved model.
    pandas.testing.assert_frame_equal(
        results_df, model_data_info.expected_results_df, check_less_precise=3
    )

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(model_data_info.inference_df):
        inp_dict[df_col_name] = model_data_info.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_dict(data=results),
        model_data_info.expected_results_df,
        check_less_precise=3,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(model_data_info.inference_df):
        inp_list.append(model_data_info.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_pyfunc_serve_and_score():
    model_data_info = save_or_log_tf_model_by_mlflow128("iris", "log_model")
    model_uri = f"runs:/{model_data_info.run_id}/model"

    resp = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=model_data_info.inference_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--env-manager", "local"],
    )
    actual = pd.DataFrame(json.loads(resp.content))["class_ids"].values
    expected = (
        model_data_info.expected_results_df["predictions"]
        .map(iris_data_utils.SPECIES.index)
        .values
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_tf_saved_model_model_with_tf_keras_api(tmpdir):
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
    save_tf_estimator_model(
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
        np.testing.assert_allclose(predictions["dense"], model.predict(feed_dict).squeeze())

    load_and_predict()


def test_saved_model_support_array_type_input():
    def infer(features):
        res = np.expand_dims(features.numpy().sum(axis=1), axis=1)
        return {"prediction": tf.constant(res)}

    model = _TF2Wrapper(None, infer)
    infer_df = pd.DataFrame({"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]})

    result = model.predict(infer_df)

    np.testing.assert_allclose(result["prediction"], infer_df.applymap(sum).values[:, 0])


def test_virtualenv_subfield_points_to_correct_path(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()
