# Location: mlflow/mlflow/tracking/fluent.py:1248
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1248 '])
def test(_):
    import mlflow

    # Case sensitive name
    experiment = mlflow.get_experiment_by_name("Default")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")


if __name__ == "__main__":
    test()
