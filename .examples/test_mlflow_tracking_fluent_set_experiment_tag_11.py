# Location: mlflow/mlflow/tracking/fluent.py:613
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:613 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        mlflow.set_experiment_tag("release.version", "2.2.0")


if __name__ == "__main__":
    test()
