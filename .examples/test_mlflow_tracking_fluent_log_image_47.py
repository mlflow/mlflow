# Location: mlflow/mlflow/tracking/fluent.py:1086
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1086 '])
def test(_):
    import mlflow
    from PIL import Image

    image = Image.new("RGB", (100, 100))

    with mlflow.start_run():
        mlflow.log_image(image, "image.png")


if __name__ == "__main__":
    test()
