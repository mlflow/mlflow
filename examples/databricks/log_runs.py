"""
Logs MLflow runs in Databricks from an external host.

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
    experiment = mlflow.set_experiment(f"/Users/{args.user}/{uuid.uuid4().hex}")

    mlflow.sklearn.autolog()
    num_runs = 5
    print(f"Logging {num_runs} runs in {args.host}#/mlflow/experiments/{experiment.experiment_id}")
    for i in range(num_runs):
        with mlflow.start_run() as run:
            print(f"Logging run:", run.info.run_id, f"{i + 1} / {num_runs} ")
            LinearRegression().fit(*load_iris(as_frame=True, return_X_y=True))


if __name__ == "__main__":
    main()
