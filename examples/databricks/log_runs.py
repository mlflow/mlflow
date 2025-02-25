"""
Logs MLflow runs in Databricks from an external host.

How to run:
$ python examples/databricks/log_runs.py --host <host> --token <token> --user <user> [--experiment-id 123]

See also:
https://docs.databricks.com/dev-tools/api/latest/authentication.html#generate-a-personal-access-token
"""

import argparse
import os
import uuid

from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, ParameterGrid

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Databricks workspace URL")
    parser.add_argument("--token", help="Databricks personal access token")
    parser.add_argument("--user", help="Databricks username")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="ID of the experiment to log runs in. If unspecified, a new experiment will be created.",
    )
    args = parser.parse_args()

    os.environ["DATABRICKS_HOST"] = args.host
    os.environ["DATABRICKS_TOKEN"] = args.token

    mlflow.set_tracking_uri("databricks")
    if args.experiment_id:
        experiment = mlflow.set_experiment(experiment_id=args.experiment_id)
    else:
        experiment = mlflow.set_experiment(f"/Users/{args.user}/{uuid.uuid4().hex}")

    print(f"Logging runs in {args.host}#/mlflow/experiments/{experiment.experiment_id}")
    mlflow.sklearn.autolog(max_tuning_runs=None)
    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 5, 10]}
    clf = GridSearchCV(svm.SVC(), parameters)
    clf.fit(iris.data, iris.target)

    # Log unnested runs
    for params in ParameterGrid(parameters):
        clf = svm.SVC(**params)
        clf.fit(iris.data, iris.target)


if __name__ == "__main__":
    main()
