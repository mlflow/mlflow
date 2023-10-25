# Location: mlflow/mlflow/tracking/fluent.py:1247
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1247 '])
def test(_):
    import mlflow

    experiment = mlflow.get_experiment("0")
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")


if __name__ == "__main__":
    test()
