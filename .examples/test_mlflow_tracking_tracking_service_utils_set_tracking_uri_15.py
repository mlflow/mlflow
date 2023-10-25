# Location: mlflow/mlflow/tracking/_tracking_service/utils.py:54
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/_tracking_service/utils.py:54 '])
def test(_):
    import mlflow

    mlflow.set_tracking_uri("file:///tmp/my_tracking")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")


if __name__ == "__main__":
    test()
