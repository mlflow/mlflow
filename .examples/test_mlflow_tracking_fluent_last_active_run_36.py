# Location: mlflow/mlflow/tracking/fluent.py:484
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:484 '])
def test(_):
    import mlflow

    mlflow.start_run()
    run = mlflow.last_active_run()
    mlflow.end_run()


if __name__ == "__main__":
    test()
