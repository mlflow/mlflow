# Location: mlflow/tracking/fluent.py:1807
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1807 '])
def test(_):
    import numpy as np
    import mlflow.sklearn
    from mlflow import MlflowClient
    from sklearn.linear_model import LinearRegression


    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))


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
