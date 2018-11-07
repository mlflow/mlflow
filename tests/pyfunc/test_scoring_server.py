import os
import json
import pandas as pd
import numpy as np
from collections import namedtuple

import pytest
import sklearn.datasets as datasets
import sklearn.neighbors as knn

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.protos.databricks_pb2 import ErrorCode, MALFORMED_REQUEST, BAD_REQUEST

from tests.helper_functions import pyfunc_serve_and_score_model


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="session")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture
def model_path(tmpdir):
    return str(os.path.join(tmpdir.strpath, "model"))


def test_scoring_server_responds_to_invalid_json_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    incorrect_json_content = json.dumps({"not": "a serialized dataframe"})
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), 
            data=incorrect_json_content, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception trace" in response_json["message"]


def test_scoring_server_responds_to_malformed_json_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), 
            data=malformed_json_content, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception trace" in response_json["message"]


def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # The pyfunc scoring server expects a serialized Pandas Dataframe in `split` or `records` 
    # format; passing a serialized Dataframe in `table` format should yield a readable error
    pandas_record_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="table")
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), 
            data=pandas_record_content, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert "error_code" in response_json
    print(response_json)
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception trace" in response_json["message"]


def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), 
            data=incompatible_df, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "Original exception trace" in response_json["message"]


def test_scoring_server_responds_to_invalid_csv_input_with_mlflow_exception_text(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # Any empty string is not valid pandas CSV
    incorrect_csv_content = ""
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), 
            data=incorrect_csv_content, 
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    print(response_json)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception trace" in response_json["message"]
