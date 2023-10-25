# Location: mlflow/mlflow/tracking/fluent.py:648
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:648 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        mlflow.set_tag("release.version", "2.2.0")


if __name__ == "__main__":
    test()
