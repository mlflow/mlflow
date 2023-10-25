# Location: mlflow/mlflow/tracking/fluent.py:731
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:731 '])
def test(_):
    import mlflow

    params = {"learning_rate": 0.01, "n_estimators": 10}

    # Log a batch of parameters
    with mlflow.start_run():
        mlflow.log_params(params)


if __name__ == "__main__":
    test()
