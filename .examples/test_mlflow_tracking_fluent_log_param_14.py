# Location: mlflow/mlflow/tracking/fluent.py:583
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:583 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        value = mlflow.log_param("learning_rate", 0.01)
        assert value == 0.01


if __name__ == "__main__":
    test()
