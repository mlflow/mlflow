from pprint import pprint

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import fetch_logged_data

import mlflow
import mlflow.xgboost


def main():
    # prepare example dataset
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    regressor = xgb.XGBRegressor(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3)
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = regressor.predict(X_test)
    mean_squared_error(y_test, y_pred)
    run_id = mlflow.last_active_run().info.run_id
    print(f"Logged data and model in run {run_id}")

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print(f"\n---------- logged {key} ----------")
        pprint(data)


if __name__ == "__main__":
    main()
