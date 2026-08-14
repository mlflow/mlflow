from pprint import pprint

import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from utils import fetch_logged_data

import mlflow


def main():
    mlflow.sklearn.autolog()

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    clf.fit(iris.data, iris.target)
    run_id = mlflow.last_active_run().info.run_id

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print(f"\n---------- logged {key} ----------")
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = f"tags.mlflow.parentRunId = '{run_id}'"
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = [f"params.{p}" for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])


if __name__ == "__main__":
    main()
