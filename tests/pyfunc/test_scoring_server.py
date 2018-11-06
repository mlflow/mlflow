import os
import json

import pytest
import sklearn.datasets as datasets
import sklearn.neighbors as knn

import mlflow.sklearn
from mlflow.protos.databricks_pb2 import ErrorCode, MALFORMED_REQUEST

from tests.helper_functions import pyfunc_serve_and_score_model


@pytest.fixture(scope="session")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return knn_model


@pytest.fixture
def model_path(tmpdir):
    return str(os.path.join(tmpdir.strpath, "model"))


def test_pyfunc_scoring_server_responds_to_invalid_json_input_with_mlflow_exception_text(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

    incorrect_json_content = json.dumps({"not": "a serialized dataframe"})
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), data=incorrect_json_content, data_type="json")
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception text" in response_json["message"]


def test_pyfunc_scoring_server_responds_to_malformed_json_input_with_mlflow_exception_text(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

    malformed_json_content = "this is,,,, not valid json"
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), data=malformed_json_content, data_type="json")
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception text" in response_json["message"]


# def test_pyfunc_scoring_server_responds_to_invalid_csv_input_with_mlflow_exception_text(
#         sklearn_model, model_path):
#     mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
#     
#     incorrect_csv_content = ",not,a,pandas,dataframe,0.1"
#     response_json = pyfunc_serve_and_score_model(
#             model_path=os.path.abspath(model_path), data=incorrect_csv_content, data_type="csv")
#     print(response_json)
#     assert "error_code" in response_json
#     assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
#     assert "message" in response_json
#     assert "Original exception text" in response_json["message"]


def test_pyfunc_scoring_server_responds_to_malformed_csv_input_with_mlflow_exception_text(
        sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    
    incorrect_csv_content = "this is not)))) valid csv"
    response_json = pyfunc_serve_and_score_model(
            model_path=os.path.abspath(model_path), data=incorrect_csv_content, data_type="csv")
    assert "error_code" in response_json
    assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    assert "message" in response_json
    assert "Original exception text" in response_json["message"]

# def test_pyfunc_scoring_server_responds_to_malformed_json_input_with_mlflow_exception_text(
    #     sklearn_model, model_path):
    # mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    #
    # incorrect_json_content = "this is,,,, not valid json"
    # response_json = pyfunc_serve_and_score_model(
    #         model_path=os.path.abspath(model_path), data=incorrect_json_content, data_type="json")
    # assert "error_code" in response_json
    # assert response_json["error_code"] == ErrorCode.Name(MALFORMED_REQUEST)
    # assert "message" in response_json
    # assert "Original exception text" in response_json["message"]
