# Location: mlflow/mlflow/tracking/fluent.py:206
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:206 '])
def test(_):
    import mlflow

    # Start run and get status
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")

    # End run and get status
    mlflow.end_run()
    run = mlflow.get_run(run.info.run_id)
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    print("--")

    # Check for any active runs
    print(f"Active run: {mlflow.active_run()}")


if __name__ == "__main__":
    test()
