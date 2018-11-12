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
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=incorrect_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_responds_to_malformed_json_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=malformed_json_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # The pyfunc scoring server expects a serialized Pandas Dataframe in `split` or `records`
    # format; passing a serialized Dataframe in `table` format should yield a readable error
    pandas_table_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="table")
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=pandas_table_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=incompatible_df,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(BAD_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_responds_to_invalid_csv_input_with_stacktrace_and_error_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    # Any empty string is not valid pandas CSV
    incorrect_csv_content = ""
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=incorrect_csv_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_CSV)
    response_json = json.loads(response.content)
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "stack_trace" in response_json


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_records_orientation(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_record_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="records")
    response_default_content_type = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=pandas_record_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON)
    assert response_default_content_type.status_code == 200

    response_records_content_type = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=pandas_record_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED)
    assert response_records_content_type.status_code == 200


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_split_orientation(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    assert response.status_code == 200


def test_scoring_server_responds_to_invalid_content_type_request_with_unsupported_content_type_code(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    pandas_split_content = pd.DataFrame(sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path),
            data=pandas_split_content,
            content_type="not_a_supported_content_type")
    assert response.status_code == 415
