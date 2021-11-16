import mlflow

from mlflow.evaluation import evaluate, EvaluationDataset
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pytest

from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def load_json_artifact(artifact_path):
    import json

    fpath = mlflow.get_artifact_uri(artifact_path).replace("file://", "")
    with open(fpath, "r") as f:
        return json.load(f)


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
def evaluation_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    return EvaluationDataset(data=eval_X, labels=eval_y, name='eval_data_1')


def test_reg_evaluate(regressor_model, evaluation_dataset):
    y_true = evaluation_dataset.labels
    y_pred = regressor_model.predict(evaluation_dataset.data)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_metrics = {
        'mean_absolute_error': expected_mae,
        'mean_squared_error': expected_mse,
    }
    expected_saved_metrics = {
        'mean_absolute_error_on_eval_data_1': expected_mae,
        'mean_squared_error_on_eval_data_1': expected_mse,
    }

    expected_artifact = expected_metrics

    with mlflow.start_run() as run:
        eval_result = evaluate(
            regressor_model, 'regressor', evaluation_dataset,
            run_id=None, evaluators='dummy_evaluator',
            evaluator_config={
                'can_evaluate': True,
                'metrics_to_calc': ['mean_absolute_error', 'mean_squared_error']
            }
        )
        saved_artifact_uri = mlflow.get_artifact_uri('metrics_artifact.json')
        saved_artifact = load_json_artifact('metrics_artifact.json')
        assert saved_artifact == expected_artifact

    _, saved_metrics, _, _ = get_run_data(run.info.run_id)
    assert saved_metrics == expected_saved_metrics

    assert eval_result.metrics == expected_metrics
    assert eval_result.artifacts.content == expected_artifact
    assert eval_result.artifacts.location == saved_artifact_uri
