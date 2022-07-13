from pprint import pprint

import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import mlflow
import mlflow.lightgbm

from utils import fetch_logged_data


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
    f1 = f1_score(y_test, y_pred, average="micro")
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()
