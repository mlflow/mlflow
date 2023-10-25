# Location: mlflow/mlflow/tracking/fluent.py:961
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:961 '])
def test(_):
    import mlflow

    dictionary = {"k": "v"}

    with mlflow.start_run():
        # Log a dictionary as a JSON file under the run's root artifact directory
        mlflow.log_dict(dictionary, "data.json")

        # Log a dictionary as a YAML file in a subdirectory of the run's root artifact directory
        mlflow.log_dict(dictionary, "dir/data.yml")

        # If the file extension doesn't exist or match any of [".json", ".yaml", ".yml"],
        # JSON format is used.
        mlflow.log_dict(dictionary, "data")
        mlflow.log_dict(dictionary, "data.txt")


if __name__ == "__main__":
    test()
