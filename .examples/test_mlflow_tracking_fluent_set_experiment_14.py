# Location: mlflow/mlflow/tracking/fluent.py:18
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:18 '])
def test(_):
    import mlflow

    # Set an experiment name, which must be unique and case-sensitive.
    experiment = mlflow.set_experiment("Social NLP Experiments")

    # Get Experiment Details
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


if __name__ == "__main__":
    test()
