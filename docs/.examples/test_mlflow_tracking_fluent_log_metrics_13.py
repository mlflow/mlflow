# Location: mlflow/tracking/fluent.py:711
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:711 '])
def test(_):
    import mlflow

    metrics = {"mse": 2500.00, "rmse": 50.00}

    # Log a batch of metrics
    with mlflow.start_run():
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    test()
