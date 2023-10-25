# Location: mlflow/mlflow/tracking/fluent.py:455
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:455 '])
def test(_):
    import mlflow

    mlflow.start_run()
    run = mlflow.active_run()
    print(f"Active run_id: {run.info.run_id}")
    mlflow.end_run()


if __name__ == "__main__":
    test()
