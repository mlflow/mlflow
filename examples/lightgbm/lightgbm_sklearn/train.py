from pprint import pprint

import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from utils import fetch_logged_data

import mlflow
import mlflow.lightgbm


def main():
    # prepare example dataset
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # enable auto logging
    # this includes lightgbm.sklearn estimators
    mlflow.lightgbm.autolog()

    regressor = lgb.LGBMClassifier(n_estimators=20, reg_lambda=1.0)
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = regressor.predict(X_test)
    f1_score(y_test, y_pred, average="micro")
    run_id = mlflow.last_active_run().info.run_id
    print(f"Logged data and model in run {run_id}")

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print(f"\n---------- logged {key} ----------")
        pprint(data)


if __name__ == "__main__":
    main()
