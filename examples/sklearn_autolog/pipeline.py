from pprint import pprint

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import fetch_logged_data

import mlflow


def main():
    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X, y)
    run_id = mlflow.last_active_run().info.run_id
    print(f"Logged data and model in run: {run_id}")

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print(f"\n---------- logged {key} ----------")
        pprint(data)


if __name__ == "__main__":
    main()
