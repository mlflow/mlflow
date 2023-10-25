# Location: mlflow/mlflow/tracking/fluent.py:1481
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1481 '])
def test(_):
    import mlflow

    experiment_id = mlflow.create_experiment("New Experiment")
    mlflow.delete_experiment(experiment_id)

    # Examine the deleted experiment details.
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Last Updated timestamp: {experiment.last_update_time}")


if __name__ == "__main__":
    test()
