# Location: mlflow/mlflow/tracking/fluent.py:779
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:779 '])
def test(_):
    import numpy as np
    import mlflow

    array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = mlflow.data.from_numpy(array, source="data.csv")

    # Log an input dataset used for training
    with mlflow.start_run():
        mlflow.log_input(dataset, context="training")


if __name__ == "__main__":
    test()
