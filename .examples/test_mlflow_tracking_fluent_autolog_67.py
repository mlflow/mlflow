# Location: mlflow/mlflow/tracking/fluent.py:1987
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1987 '])
def test(_):
    import numpy as np
    import mlflow.sklearn
    from mlflow import MlflowClient
    from sklearn.linear_model import LinearRegression


    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print(f"run_id: {r.info.run_id}")
        print(f"artifacts: {artifacts}")
        print(f"params: {r.data.params}")
        print(f"metrics: {r.data.metrics}")
        print(f"tags: {tags}")


    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # Auto log all the parameters, metrics, and artifacts
    mlflow.autolog()
    model = LinearRegression()
    with mlflow.start_run() as run:
        model.fit(X, y)

    # fetch the auto logged parameters and metrics for ended run
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    test()
