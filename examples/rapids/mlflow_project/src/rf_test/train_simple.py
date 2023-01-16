"""Simple example integrating cuML with MLFlow"""

import argparse
from functools import partial

import mlflow
import mlflow.sklearn

from cuml.metrics.accuracy import accuracy_score
from cuml.preprocessing.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier


def load_data(fpath):
    """
    Simple helper function for loading data to be used by CPU/GPU models.

    :param fpath: Path to the data to be ingested
    :return: DataFrame wrapping the data at [fpath]. Data will be in either a Pandas or RAPIDS (cuDF) DataFrame
    """
    import cudf

    df = cudf.read_parquet(fpath)
    X = df.drop(["ArrDelayBinary"], axis=1)
    y = df["ArrDelayBinary"].astype("int32")

    return train_test_split(X, y, test_size=0.2)


def train(fpath, max_depth, max_features, n_estimators):
    """
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param max_depth: RF max_depth parameter
    :param max_features: RF max_features parameter
    :param n_estimators: RF n_estimators parameter
    :return: trained model
    """
    X_train, X_test, y_train, y_test = load_data(fpath)

    mod = RandomForestClassifier(
        max_depth=max_depth, max_features=max_features, n_estimators=n_estimators
    )

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlparams = {
        "max_depth": str(max_depth),
        "max_features": str(max_features),
        "n_estimators": str(n_estimators),
    }
    mlflow.log_params(mlparams)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(mod, "saved_models")

    return mod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="tpe", choices=["tpe"], type=str)
    parser.add_argument("--conda-env", required=True, type=str)
    parser.add_argument("--fpath", required=True, type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--max_depth", type=int, default=12)

    args = parser.parse_args()

    experid = 0

    artifact_path = "Airline-Demo"
    artifact_uri = None
    experiment_name = "RAPIDS-CLI"
    experiment_id = None

    mlflow.set_tracking_uri(uri="sqlite:////tmp/mlflow-db.sqlite")
    with mlflow.start_run(run_name="RAPIDS-MLFlow"):
        model = train(args.fpath, args.max_depth, args.max_features, args.n_estimators)

        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name="rapids_mlflow_cli",
            conda_env="conda.yaml",
        )
        artifact_uri = mlflow.get_artifact_uri(artifact_path=artifact_path)

    print("Model uri: %s" % artifact_uri)
