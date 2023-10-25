# Location: mlflow/mlflow/data/numpy_dataset.py:197
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/data/numpy_dataset.py:197 '])
def test(_):
    import mlflow
    import numpy as np

    x = {
        "feature_1": np.random.uniform(size=[2, 5, 4]),
        "feature_2": np.random.uniform(size=[2, 5, 4]),
    }
    y = np.random.randint(2, size=[2])
    dataset = mlflow.data.from_numpy(x, targets=y)


if __name__ == "__main__":
    test()
