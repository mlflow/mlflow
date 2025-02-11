# Location: mlflow/tracking/_tracking_service/utils.py:63
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/_tracking_service/utils.py:63 '])
def test(_):
    import mlflow

    mlflow.set_tracking_uri("file:///tmp/my_tracking")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))


if __name__ == "__main__":
    test()
