from pprint import pprint

import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import mlflow
import mlflow.xgboost

from utils import fetch_logged_data


def main():
    # prepare example dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    with mlflow.start_run() as run:

        regressor = xgb.XGBRegressor(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        run_id = run.info.run_id
        print("Logged data and model in run {}".format(run_id))
        mlflow.xgboost.log_model(regressor, artifact_path="log_model")

    # show logged data
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    mlflow.xgboost.save_model(regressor, "trained_model/")
    reload_model = mlflow.pyfunc.load_model("trained_model/")
    np.testing.assert_array_almost_equal(y_pred, reload_model.predict(X_test))


if __name__ == "__main__":
    main()
