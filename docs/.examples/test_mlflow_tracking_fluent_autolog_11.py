# Location: mlflow/tracking/fluent.py:1752
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1752 '])
def test(_):
    import mlflow

    mlflow.autolog(log_models=False, exclusive=True)
    import sklearn


if __name__ == "__main__":
    test()
