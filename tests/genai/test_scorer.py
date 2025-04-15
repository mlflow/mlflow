from collections import namedtuple

import mlflow
from mlflow import MlflowClient

RunData = namedtuple("RunData", ["params", "metrics", "tags", "artifacts"])


def get_run_data(run_id):
    # TODO: not sure if this is really needed
    client = MlflowClient()
    data = client.get_run(run_id).data
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return RunData(params=data.params, metrics=data.metrics, tags=data.tags, artifacts=artifacts)


def test_scorer_existence_in_run():
    with mlflow.start_run() as _run:
        result = mlflow.evaluate(
            # TODO: sample data
            model_type="databricks-agent",
        )

    assert any("scorer_name" in metric_name for metric_name in result.metrics.keys())

    # _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    # model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    # y = diabetes_dataset.labels_data
    # y_pred = model.predict(diabetes_dataset.features_data)

    # expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)
    # expected_metrics["score"] = model._model_impl.score(
    #     diabetes_dataset.features_data, diabetes_dataset.labels_data
    # )

    # assert_metrics_equal(result.metrics, expected_metrics)
    # assert "mlflow.datassets" not in tags


"""
Other tests:
1. check whether incorrect signatures are allowed in decorator
2. check that individual scorers are being used properly by checking their outputs on per-row level
3. check that all parameters are used in the decorated function (input, output, expected, trace)
4. check if aggregation parameters are used in the decorated function by checking expected values
5. check whether the scorer name is set correctly in decorated function
6. check that non-decorated/manually inherited class also works properly
"""
