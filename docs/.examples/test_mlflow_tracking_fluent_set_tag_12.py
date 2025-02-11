# Location: mlflow/tracking/fluent.py:633
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:633 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        mlflow.set_tag("release.version", "2.2.0")


if __name__ == "__main__":
    test()
