"""Hyperparameter optimization with cuML, hyperopt, and MLflow"""

import argparse
from functools import partial

from cuml.ensemble import RandomForestClassifier
from cuml.metrics.accuracy import accuracy_score
from cuml.preprocessing.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

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


def _train(params, fpath, hyperopt=False):
    """
    Args:
        params: Hyperparameters. Its structure is consistent with how search space is defined.
        fpath: Path or URL for the training data used with the model.
        hyperopt: Use hyperopt for hyperparameter search during training.

    Returns:
        dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run).
    """
    max_depth, max_features, n_estimators = params
    max_depth = int(max_depth)
    max_features = float(max_features)
    n_estimators = int(n_estimators)

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
    signature = infer_signature(X_train, predictions)

    mlflow.sklearn.log_model(mod, name="saved_models", signature=signature)

    if not hyperopt:
        return mod

    return {"loss": acc, "status": STATUS_OK}


def train(params, fpath, hyperopt=False):
    """
    Proxy function used to call _train

    Args:
        params: Hyperparameters. Its structure is consistent with how search space is defined.
        fpath: Path or URL for the training data used with the model.
        hyperopt: Use hyperopt for hyperparameter search during training.

    Returns:
        dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)

    """
    with mlflow.start_run(nested=True):
        return _train(params, fpath, hyperopt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="tpe", choices=["tpe"], type=str)
    parser.add_argument("--conda-env", required=True, type=str)
    parser.add_argument("--fpath", required=True, type=str)
    args = parser.parse_args()

    search_space = [
        hp.uniform("max_depth", 5, 20),
        hp.uniform("max_features", 0.1, 1.0),
        hp.uniform("n_estimators", 150, 1000),
    ]

    trials = Trials()
    algorithm = tpe.suggest if args.algo == "tpe" else None
    fn = partial(train, fpath=args.fpath, hyperopt=True)
    experid = 0

    artifact_path = "Airline-Demo"
    artifact_uri = None

    mlflow.set_tracking_uri(uri="sqlite:////tmp/mlflow-db.sqlite")
    with mlflow.start_run(run_name="RAPIDS-Hyperopt"):
        argmin = fmin(fn=fn, space=search_space, algo=algorithm, max_evals=2, trials=trials)

        print("===========")
        fn = partial(train, fpath=args.fpath, hyperopt=False)
        final_model = fn(tuple(argmin.values()))

        mlflow.sklearn.log_model(
            final_model,
            name=artifact_path,
            registered_model_name="rapids_mlflow_cli",
            conda_env="envs/conda.yaml",
        )
