import mlflow

from mlflow.evaluation import evaluate, EvaluationDataset
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pytest
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from mlflow_test_plugin.dummy_evaluator import Array2DEvaluationArtifact
from mlflow.tracking.artifact_utils import get_artifact_uri


def get_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
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


def test_classifier_evaluate(classifier_model, iris_dataset):
    y_true = iris_dataset.labels
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
            evaluator_config={"can_evaluate": True,},
        )

    artifact_name = "confusion_matrix_on_iris_dataset.csv"
    saved_artifact_path = get_local_artifact_path(run.info.run_id, artifact_name)

    _, saved_metrics, _, saved_artifacts = get_run_data(run.info.run_id)
    assert saved_metrics == expected_saved_metrics
    assert saved_artifacts == [artifact_name]

    assert eval_result.metrics == expected_metrics
    returned_confusion_matrix_artifact = eval_result.artifacts[artifact_name]
    assert np.array_equal(returned_confusion_matrix_artifact.content, expected_artifact)
    assert np.array_equal(
        Array2DEvaluationArtifact.load_content_from_file(saved_artifact_path), expected_artifact
    )
    assert returned_confusion_matrix_artifact.location == get_artifact_uri(
        run.info.run_id, artifact_name
    )


def test_regressor_evaluate(regressor_model, iris_dataset):
    y_true = iris_dataset.labels
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
    with mlflow.start_run() as run:
        eval_result = evaluate(
            regressor_model,
            "regressor",
            iris_dataset,
            run_id=None,
            evaluators="dummy_evaluator",
            evaluator_config={"can_evaluate": True,},
        )
    _, saved_metrics, _, _ = get_run_data(run.info.run_id)
    assert saved_metrics == expected_saved_metrics

    assert eval_result.metrics == expected_metrics
