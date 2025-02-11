# Location: mlflow/tracking/fluent.py:466
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:466 '])
def test(_):
    import mlflow

    mlflow.start_run()
    mlflow.end_run()
    run = mlflow.last_active_run()


if __name__ == "__main__":
    test()
