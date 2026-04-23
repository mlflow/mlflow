import inspect

import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

NUM_RUNS = 1005


def main() -> None:
    print("mlflow from:", mlflow.__file__)
    print("search_runs signature:", inspect.signature(MlflowClient.search_runs))
    tracking_dir = Path(tempfile.mkdtemp(prefix="mlflow-search-runs-warning-"))
    mlflow.set_tracking_uri(tracking_dir.as_uri())

    experiment_id = mlflow.create_experiment("search-runs-warning-repro")
    client = MlflowClient()

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Creating {NUM_RUNS} runs...")

    for i in range(NUM_RUNS):
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("i", i)

    print()
    print("Calling MlflowClient.search_runs() with the default max_results...")
    client_page = client.search_runs(experiment_ids=[experiment_id])
    print(f"len(page) = {len(client_page)}")
    print(f"next_page_token = {client_page.token!r}")

    print()
    print("Calling mlflow.search_runs() for comparison...")
    fluent_runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
    print(f"len(runs) = {len(fluent_runs)}")


if __name__ == "__main__":
    main()
