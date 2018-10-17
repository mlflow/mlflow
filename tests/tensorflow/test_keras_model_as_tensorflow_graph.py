"""
AWS Sagemaker converts keras models to tensorflow graph on saving.
In order to process 2D input tensors in such models, pandas duplicated column
names should be allowed.
"""

import os

import mlflow.tensorflow
from mlflow import pyfunc
import pandas as pd


def test_keras_boston_housing_scoring_server_csv_prediction(tmpdir, boston_housing_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=boston_housing_model["path"],
                                 tf_meta_graph_tags=boston_housing_model["meta_graph_tags"],
                                 tf_signature_def_key=boston_housing_model["signature_def_key"],
                                 path=model_path)
    pyfunc_wrapper = pyfunc.load_pyfunc(model_path)

    df = pd.DataFrame(
        boston_housing_model["test_data"],
        columns=[boston_housing_model["input_tensor_name"]] *
        boston_housing_model["test_data"].shape[1])

    test_data_path = os.path.join(str(tmpdir), "test_data.csv")
    df.to_csv(test_data_path, index=False)

    fp = open(test_data_path)
    data = pd.read_csv(fp, header=None)  # HACK https://github.com/pandas-dev/pandas/issues/19383
    data = data.rename(columns=data.iloc[0], copy=False).iloc[1:].reset_index(drop=True)

    predicts = pyfunc_wrapper.predict(data)

    assert predicts.shape[0] == boston_housing_model["test_data"].shape[0]
    assert predicts.shape[1] == 1

    predicts_list = predicts[boston_housing_model["output_tensor_name"]].tolist()
    assert len(list(filter(None, predicts_list))) == predicts.shape[0]


def test_keras_boston_housing_scoring_server_json_prediction(tmpdir, boston_housing_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=boston_housing_model["path"],
                                 tf_meta_graph_tags=boston_housing_model["meta_graph_tags"],
                                 tf_signature_def_key=boston_housing_model["signature_def_key"],
                                 path=model_path)
    pyfunc_wrapper = pyfunc.load_pyfunc(model_path)

    df = pd.DataFrame(
        boston_housing_model["test_data"],
        columns=[boston_housing_model["input_tensor_name"]] *
        boston_housing_model["test_data"].shape[1])

    test_data_path = os.path.join(str(tmpdir), "test_data.json")
    df.to_json(test_data_path, orient="split")

    fp = open(test_data_path)
    data = pd.read_json(fp, orient="split")

    predicts = pyfunc_wrapper.predict(data)

    assert predicts.shape[0] == boston_housing_model["test_data"].shape[0]
    assert predicts.shape[1] == 1

    predicts_list = predicts[boston_housing_model["output_tensor_name"]].tolist()
    assert len(list(filter(None, predicts_list))) == predicts.shape[0]
