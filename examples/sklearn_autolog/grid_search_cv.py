from pprint import pprint

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import mlflow


def yield_artifacts(run_id, path=None):
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }


def main():
    mlflow.sklearn.autolog()

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    with mlflow.start_run() as run:
        clf.fit(iris.data, iris.target)

    # show data logged in the parent run:
    print("parent run")
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    print("\nchild runs")
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]
    print(runs[["run_id", *param_cols, *metric_cols]])


if __name__ == "__main__":
    main()
