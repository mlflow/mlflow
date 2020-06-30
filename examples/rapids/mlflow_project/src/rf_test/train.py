"""Hyperparameter optimization with cuML, hyperopt, and MLFlow"""

import argparse
from functools import partial

import mlflow
import mlflow.sklearn

from cuml.metrics.accuracy import accuracy_score
from cuml.preprocessing.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def load_data(fpath):
    """
    Simple helper function for loading data to be used by CPU/GPU models.

    :param fpath: Path to the data to be ingested
    :return: DataFrame wrapping the data at [fpath]. Data will be in either a Pandas or RAPIDS (cuDF) DataFrame
    """
    import cudf

    df = cudf.read_parquet(fpath)
    X = df.drop(["ArrDelayBinary"], axis=1)
    y = df["ArrDelayBinary"].astype('int32')

    return train_test_split(X, y, test_size=0.2)


def _train(params, fpath, hyperopt=False):
    """
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param hyperopt: Use hyperopt for hyperparameter search during training.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    max_depth, max_features, n_estimators = params
    max_depth, max_features, n_estimators = (int(max_depth), float(max_features), int(n_estimators))

    X_train, X_test, y_train, y_test = load_data(fpath)

    mod = RandomForestClassifier(max_depth=max_depth,
                                 max_features=max_features,
                                 n_estimators=n_estimators)

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlparams = {"max_depth": str(max_depth),
                "max_features": str(max_features),
                "n_estimators": str(n_estimators)}
    mlflow.log_params(mlparams)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(mod, "saved_models")

    if (not hyperopt):
        return mod

    return {'loss': acc, 'status': STATUS_OK}


def train(params, fpath, hyperopt=False):
    """
    Proxy function used to call _train
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param hyperopt: Use hyperopt for hyperparameter search during training.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    with mlflow.start_run(nested=True):
        return _train(params, fpath, hyperopt)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='tpe', choices=['tpe'], type=str)
    parser.add_argument('--conda-env', required=True, type=str)
    parser.add_argument('--fpath', required=True, type=str)
    args = parser.parse_args()

    search_space = [
        hp.uniform('max_depth', 5, 20),
        hp.uniform('max_features', 0.1, 1.0),
        hp.uniform('n_estimators', 150, 1000)
    ]

    trials = Trials()
    algorithm = tpe.suggest if args.algo == 'tpe' else None
    fn = partial(train, fpath=args.fpath, hyperopt=True)
    experid = 0

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "RAPIDS-Hyperopt")
        argmin = fmin(fn=fn,
                      space=search_space,
                      algo=algorithm,
                      max_evals=2,
                      trials=trials)

        print("===========")
        fn = partial(train, fpath=args.fpath, hyperopt=False)
        final_model = fn(tuple(argmin.values()))

        conda_data = ""
        if (args.conda_env.startswith("http")):
            import requests

            resp = requests.get(args.conda_env)
            conda_data = str(resp.text)
        else:
            with open(args.conda_env, 'r') as reader:
                conda_data = reader.read()

        with open("conda.yaml", 'w') as writer:
            writer.write(conda_data)

        mlflow.sklearn.log_model(final_model, "rapids_mlflow", conda_env='conda.yaml')
