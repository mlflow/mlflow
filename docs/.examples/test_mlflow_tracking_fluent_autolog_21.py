# Location: mlflow/tracking/fluent.py:1762
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1762 '])
def test(_):
    import mlflow

    mlflow.autolog(log_models=False, exclusive=True)

    import sklearn

    mlflow.sklearn.autolog(log_models=True)


if __name__ == "__main__":
    test()
