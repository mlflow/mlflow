import mlflow

from mlflow.models.evaluation import evaluate, EvaluationDataset
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pytest
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from mlflow.tracking.artifact_utils import get_artifact_uri
import json


def get_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items()}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return data.params, data.metrics, tags, artifacts


def get_local_artifact_path(run_id, artifact_path):
    return get_artifact_uri(run_id, artifact_path).replace("file://", "")


@pytest.fixture(scope="module")
def regressor_model():
    X, y = get_iris()
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(X, y)
    return reg


@pytest.fixture(scope="module")
def classifier_model():
    X, y = get_iris()
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="module")
def iris_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    return EvaluationDataset(data=eval_X, labels=eval_y, name="iris_dataset")


@pytest.fixture(scope="module")
def iris_pandas_df_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    data = pd.DataFrame({"f1": eval_X[:, 0], "f2": eval_X[:, 1], "y": eval_y})
    labels = "y"
    return EvaluationDataset(data=data, labels=labels, name="iris_pandas_df_dataset")


def test_classifier_evaluate(classifier_model, iris_dataset):
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(classifier_model, "clf_model")
        classifier_model_uri = get_artifact_uri(run.info.run_id, "clf_model")

    y_true = iris_dataset.labels
    classifier_model = mlflow.pyfunc.load_model(classifier_model_uri)
    y_pred = classifier_model.predict(iris_dataset.data)
    expected_accuracy_score = accuracy_score(y_true, y_pred)
    expected_metrics = {
        "accuracy_score": expected_accuracy_score,
    }
    expected_saved_metrics = {
        "accuracy_score_on_iris_dataset": expected_accuracy_score,
    }

    expected_artifact = confusion_matrix(y_true, y_pred)

    with mlflow.start_run() as run:
        eval_result = evaluate(
            classifier_model,
            "classifier",
            iris_dataset,
            run_id=None,
            evaluators="dummy_evaluator",
        )

    artifact_name = "confusion_matrix_on_iris_dataset.csv"
    saved_artifact_path = get_local_artifact_path(run.info.run_id, artifact_name)

    _, saved_metrics, _, saved_artifacts = get_run_data(run.info.run_id)
    assert saved_metrics == expected_saved_metrics
    assert saved_artifacts == [artifact_name]

    assert eval_result.metrics == expected_metrics
    returned_confusion_matrix_artifact = eval_result.artifacts[artifact_name]
    assert np.array_equal(returned_confusion_matrix_artifact.content, expected_artifact)
    assert returned_confusion_matrix_artifact.uri == get_artifact_uri(
        run.info.run_id, artifact_name
    )
    assert np.array_equal(
        returned_confusion_matrix_artifact.load(saved_artifact_path), expected_artifact
    )


def test_regressor_evaluate(regressor_model, iris_dataset):
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(regressor_model, "reg_model")
        regressor_model_uri = get_artifact_uri(run.info.run_id, "reg_model")

    y_true = iris_dataset.labels
    regressor_model = mlflow.pyfunc.load_model(regressor_model_uri)
    y_pred = regressor_model.predict(iris_dataset.data)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_metrics = {
        "mean_absolute_error": expected_mae,
        "mean_squared_error": expected_mse,
    }
    expected_saved_metrics = {
        "mean_absolute_error_on_iris_dataset": expected_mae,
        "mean_squared_error_on_iris_dataset": expected_mse,
    }

    for model in [regressor_model, regressor_model_uri]:
        with mlflow.start_run() as run:
            eval_result = evaluate(
                model,
                "regressor",
                iris_dataset,
                run_id=None,
                evaluators="dummy_evaluator",
            )
        _, saved_metrics, _, _ = get_run_data(run.info.run_id)
        assert saved_metrics == expected_saved_metrics
        assert eval_result.metrics == expected_metrics


def test_dataset_name():
    X, y = get_iris()
    d1 = EvaluationDataset(data=X, labels=y, name="a1")
    assert d1.name == "a1"
    d2 = EvaluationDataset(data=X, labels=y)
    d2.name == d2.hash


def test_dataset_hash(iris_dataset, iris_pandas_df_dataset):
    assert iris_dataset.hash == "49a04c127e5441e0f27e63a325b5fa69"
    assert iris_pandas_df_dataset.hash == "d6770fd5fffe651cb95e965854920df9"


def test_log_dataset_tag(iris_dataset, iris_pandas_df_dataset):
    with mlflow.start_run() as run:
        client = mlflow.tracking.MlflowClient()
        iris_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [iris_dataset._metadata]

        # Test appending dataset tag
        iris_pandas_df_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [
            iris_dataset._metadata,
            iris_pandas_df_dataset._metadata,
        ]

        # Test log repetitive dataset
        iris_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [
            iris_dataset._metadata,
            iris_pandas_df_dataset._metadata,
        ]
