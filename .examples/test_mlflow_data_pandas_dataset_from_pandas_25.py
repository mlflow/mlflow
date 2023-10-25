# Location: mlflow/mlflow/data/pandas_dataset.py:199
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/data/pandas_dataset.py:199 '])
def test(_):
    import mlflow
    import pandas as pd

    x = pd.DataFrame(
        [["tom", 10, 1, 1], ["nick", 15, 0, 1], ["juli", 14, 1, 1]],
        columns=["Name", "Age", "Label", "ModelOutput"],
    )
    dataset = mlflow.data.from_pandas(x, targets="Label", predictions="ModelOutput")


if __name__ == "__main__":
    test()
