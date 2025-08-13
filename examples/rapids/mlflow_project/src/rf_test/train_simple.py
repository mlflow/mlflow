"""Simple example integrating cuML with MLflow"""

import argparse

from cuml.ensemble import RandomForestClassifier
from cuml.metrics.accuracy import accuracy_score
from cuml.preprocessing.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def load_data(fpath):
    """
    Simple helper function for loading data to be used by CPU/GPU models.

    Args:
        fpath: Path to the data to be ingested

    Returns:
        DataFrame wrapping the data at [fpath]. Data will be in either a Pandas or RAPIDS (cuDF) DataFrame
    """
    import cudf

    df = cudf.read_csv(fpath)
    X = df.drop(["target"], axis=1)
    y = df["target"].astype("int32")

    return train_test_split(X, y, test_size=0.2)


def train(fpath, max_depth, max_features, n_estimators):
    """
    Train a Random Forest classifier with the specified hyperparameters.

    Args:
        fpath: Path or URL for the training data used with the model.
        max_depth: Maximum depth of the trees in the Random Forest.
        max_features: Number of features to consider when looking for the best split.
        n_estimators: Number of trees in the Random Forest.

    Returns:
        A tuple containing the trained Random Forest model and the inferred model signature.
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

    predictions = mod.predict(X_train)
    sig = infer_signature(X_train, predictions)

    mlflow.sklearn.log_model(mod, name="saved_models", signature=sig)

    return mod, sig


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
    with mlflow.start_run(run_name="RAPIDS-MLflow"):
        model, signature = train(args.fpath, args.max_depth, args.max_features, args.n_estimators)

        mlflow.sklearn.log_model(
            model,
            name=artifact_path,
            registered_model_name="rapids_mlflow_cli",
            conda_env="conda.yaml",
            signature=signature,
        )
        artifact_uri = mlflow.get_artifact_uri(artifact_path=artifact_path)

    print(f"Model uri: {artifact_uri}")
