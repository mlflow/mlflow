from hpsklearn import HyperoptEstimator, any_regressor
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import sys

import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == '__main__':
    wine_path = sys.argv[1]
    import os
    print("file =", os.path.abspath(wine_path))
    max_evals = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    trial_timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0  # default timeout is 5 mins
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 97531

    with mlflow.start_run():
        data = pd.read_csv(wine_path)
        # Create the estimator object
        estim = HyperoptEstimator(regressor=any_regressor("RoboTaster"),
                                  seed=seed,
                                  max_evals=max_evals,
                                  trial_timeout=trial_timeout)
        train, test = train_test_split(data, random_state=seed)
        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
        # Search the space of classifiers and preprocessing steps and their
        # respective hyperparameters in sklearn to fit a model to the data
        estim.fit(train_x.values, train_y.values)
        # Make a prediction using the optimized model
        predicted_qualities = estim.predict(test_x)
        # Report the accuracy of the classifier on a given set of data
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        print("best model", estim.best_model())
        mlflow.sklearn.log_model(estim, "model")

