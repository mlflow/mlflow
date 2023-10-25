# Location: mlflow/mlflow/tracking/fluent.py:1075
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1075 '])
def test(_):
    import mlflow
    import numpy as np

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, "image.png")


if __name__ == "__main__":
    test()
