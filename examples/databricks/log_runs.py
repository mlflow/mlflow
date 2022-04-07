"""
Logs MLflow runs in Databricks.

How to run:
$ python examples/databricks/log_runs.py --host <host> --token <token> --user <user>

See also:
https://docs.databricks.com/dev-tools/api/latest/authentication.html#generate-a-personal-access-token
"""
import os
import uuid
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--token")
    parser.add_argument("--user")
    args = parser.parse_args()

    os.environ["DATABRICKS_HOST"] = args.host
    os.environ["DATABRICKS_TOKEN"] = args.token

    mlflow.set_tracking_uri("databricks")
    experiment_name = f"/Users/harutaka.kawamura@databricks.com/{uuid.uuid4().hex}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    print(f"Logging runs in {args.host}#/mlflow/experiments/{experiment_id}")
    mlflow.sklearn.autolog()
    num_runs = 5
    for i in range(num_runs):
        with mlflow.start_run() as run:
            print(f"Logging run:", run.info.run_id, f"{i + 1} / {num_runs} ")
            LinearRegression().fit(*load_iris(as_frame=True, return_X_y=True))


if __name__ == "__main__":
    main()
