import collections
import json
import os

import iris_data_utils
import numpy as np
import pandas as pd
import pandas.testing
import pytest
import tensorflow as tf

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tensorflow import _TF2Wrapper
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.environment import _mlflow_conda_env

from tests.helper_functions import (
    assert_array_almost_equal,
    pyfunc_serve_and_score_model,
)


@pytest.fixture
def tf_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "pytest"])
    return conda_env


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


ModelDataInfo = collections.namedtuple(
    "ModelDataInfo",
    [
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
        "run_id",
    ],
)


def save_or_log_tf_model_by_mlflow128(tmp_path, model_type, task_type, save_path=None):
    tf_tests_dir = os.path.dirname(__file__)
    conda_env = get_or_create_conda_env(os.path.join(tf_tests_dir, "mlflow-128-tf-26-env.yaml"))
    output_data_file_path = os.path.join(tmp_path, "output_data.pkl")
    tracking_uri = mlflow.get_tracking_uri()
    exec_py_path = os.path.join(tf_tests_dir, "save_tf_estimator_model.py")
    mlflow_repo_path = os.path.dirname(os.path.dirname(tf_tests_dir))

    conda_env.execute(
        f"python {exec_py_path} "
        f"--tracking_uri {tracking_uri} "
        f"--mlflow_repo_path {mlflow_repo_path} "
        f"--model_type {model_type} "
        f"--task_type {task_type} "
        f"--save_path {save_path if save_path else 'none'}",
    )
    return ModelDataInfo(*pd.read_pickle(output_data_file_path))


def test_load_model_from_remote_uri_succeeds(tmp_path, model_path, mock_s3_bucket, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(tmp_path, "iris", "save_model", model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
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
            np.array(pred_dict[col], dtype=float),
            np.array(model_data_info.raw_results[col], dtype=float),
            rtol=1e-6,
        )


def test_iris_model_can_be_loaded_and_evaluated_successfully(tmp_path, model_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(tmp_path, "iris", "save_model", model_path)

    infer = mlflow.tensorflow.load_model(model_uri=model_path)
    feed_dict = {
        df_column_name: tf.constant(model_data_info.inference_df[df_column_name])
        for df_column_name in list(model_data_info.inference_df)
    }
    raw_preds = infer(**feed_dict)
    pred_dict = {column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()}
    for key in pred_dict.keys():
        assert_array_almost_equal(pred_dict[key], model_data_info.raw_results[key])


def test_log_and_load_model_persists_and_restores_model_successfully(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(tmp_path, "iris", "log_model")
    model_uri = f"runs:/{model_data_info.run_id}/model"
    mlflow.tensorflow.load_model(model_uri=model_uri)


def test_iris_data_model_can_be_loaded_and_evaluated_as_pyfunc(tmp_path, model_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(tmp_path, "iris", "save_model", model_path)

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(model_data_info.inference_df)
    assert isinstance(results_df, pd.DataFrame)
    pd.testing.assert_frame_equal(
        results_df.sort_index(axis=1), model_data_info.raw_df.sort_index(axis=1), rtol=1e-6
    )

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(model_data_info.inference_df):
        inp_dict[df_col_name] = model_data_info.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    pd.testing.assert_frame_equal(
        pd.DataFrame(results).sort_index(axis=1),
        model_data_info.raw_df.sort_index(axis=1),
        rtol=1e-6,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(model_data_info.inference_df):
        inp_list.append(model_data_info.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_categorical_model_can_be_loaded_and_evaluated_as_pyfunc(tmp_path, model_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(
        tmp_path, "categorical", "save_model", model_path
    )

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(model_data_info.inference_df)
    # Precision is less accurate for the categorical model when we load back the saved model.
    pandas.testing.assert_frame_equal(
        results_df.sort_index(axis=1),
        model_data_info.expected_results_df.sort_index(axis=1),
        atol=1e-5,
        rtol=0,
    )

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(model_data_info.inference_df):
        inp_dict[df_col_name] = model_data_info.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_dict(data=results).sort_index(axis=1),
        model_data_info.expected_results_df.sort_index(axis=1),
        atol=1e-5,
        rtol=0,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(model_data_info.inference_df):
        inp_list.append(model_data_info.inference_df[df_col_name].values)
    with pytest.raises(TypeError, match="Only dict and DataFrame input types are supported"):
        results = pyfunc_wrapper.predict(inp_list)


def test_pyfunc_serve_and_score(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_tf_model_by_mlflow128(tmp_path, "iris", "log_model")
    model_uri = f"runs:/{model_data_info.run_id}/model"

    resp = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=model_data_info.inference_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    actual = pd.DataFrame(json.loads(resp.content)["predictions"])["class_ids"].values
    expected = (
        model_data_info.expected_results_df["predictions"].map(iris_data_utils.SPECIES.index).values
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_saved_model_support_array_type_input():
    def infer(features):
        res = np.expand_dims(features.numpy().sum(axis=1), axis=1)
        return {"prediction": tf.constant(res)}

    model = _TF2Wrapper(None, infer)
    infer_df = pd.DataFrame({"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]})

    result = model.predict(infer_df)

    np.testing.assert_allclose(result["prediction"], infer_df.applymap(sum).values[:, 0])
