from pprint import pprint

import numpy as np
from sklearn.linear_model import LinearRegression

import mlflow


def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return {
        "keys": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }


def main():
    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    model = LinearRegression()
    with mlflow.start_run() as run:
        model.fit(X, y)

    # show logged data
    for key, data in fetch_logged_data(run._info.run_id).items():
        print("-" * 10, key, "-" * 10)
        pprint(data)


if __name__ == "__main__":
    main()
