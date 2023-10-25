# Location: mlflow/mlflow/tracking/fluent.py:917
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:917 '])
def test(_):
    import mlflow

    with mlflow.start_run():
        # Log text to a file under the run's root artifact directory
        mlflow.log_text("text1", "file1.txt")

        # Log text in a subdirectory of the run's root artifact directory
        mlflow.log_text("text2", "dir/file2.txt")

        # Log HTML text
        mlflow.log_text("<h1>header</h1>", "index.html")


if __name__ == "__main__":
    test()
