# Location: mlflow/tracking/fluent.py:685
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:685 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        mlflow.log_metric("mse", 2500.00)


if __name__ == "__main__":
    test()
