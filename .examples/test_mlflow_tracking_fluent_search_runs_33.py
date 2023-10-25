# Location: mlflow/mlflow/tracking/fluent.py:1606
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1606 '])
def test(_):
    import mlflow

    # Create an experiment and log two runs under it
    experiment_name = "Social NLP Experiments"
    experiment_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 1.55)
        mlflow.set_tag("s.release", "1.1.0-RC")
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 2.50)
        mlflow.set_tag("s.release", "1.2.0-GA")

    # Search for all the runs in the experiment with the given experiment ID
    df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    print("--")

    # Search the experiment_id using a filter_string with tag
    # that has a case insensitive pattern
    filter_string = "tags.s.release ILIKE '%rc%'"
    df = mlflow.search_runs([experiment_id], filter_string=filter_string)
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    print("--")

    # Search for all the runs in the experiment with the given experiment name
    df = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.m DESC"])
    print(df[["metrics.m", "tags.s.release", "run_id"]])


if __name__ == "__main__":
    test()
