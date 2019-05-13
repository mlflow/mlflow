import os
import pandas as pd
import pytest
import sklearn
import sklearn.datasets
import sklearn.neighbors
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import pyfunc_serve_and_score_model

@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sk_model(iris_data):
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


def test_predict(iris_data, sk_model):
    with mlflow.start_run() as active_run:
        mlflow.sklearn.log_model(sk_model, "model")
    model_uri = "runs:/{run_id}/model".format(run_id=active_run.info.run_id)
    with TempDir() as tmp:
        input_json_path = tmp.path("input.json")
        input_csv_path = tmp.path("input.csv")
        output_json_path = tmp.path("output.json")
        x, y = iris_data
        pd.DataFrame(x).to_json(input_json_path, orient="split")
        pd.DataFrame(x).to_csv(input_csv_path, index=False)
        p = subprocess.Popen(["mlflow", "models", "predict", "-m", model_uri, "-i", input_json_path,
                              "-o", output_json_path, "--no-conda"])
        p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # with conda this time
        p = subprocess.Popen(["mlflow", "models", "predict", "-m", model_uri, "-i", input_json_path,
                              "-o", output_json_path])
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with default orient (should be split)
        p = subprocess.Popen(["mlflow", "models", "predict", "-m", model_uri, "-i", input_json_path,
                              "-o", output_json_path, "-t", "json"])
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)

        # explicit json format with orient==split
        p = subprocess.Popen(["mlflow", "models", "predict", "-m", model_uri, "-i", input_json_path,
                              "-o", output_json_path, "-t", "json,split"])
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)
        # csv
        p = subprocess.Popen(["mlflow", "models", "predict", "-m", model_uri, "-i", input_csv_path,
                              "-o", output_json_path, "-t", "csv"])
        assert 0 == p.wait()
        actual = pd.read_json(output_json_path, orient="records")
        actual = actual[actual.columns[0]].values
        expected = sk_model.predict(x)
        assert all(expected == actual)








