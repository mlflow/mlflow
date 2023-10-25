# Location: mlflow/mlflow/tracking/fluent.py:1392
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1392 '])
def test(_):
    import mlflow
    from pathlib import Path

    # Create an experiment name, which must be unique and case sensitive
    experiment_id = mlflow.create_experiment(
        "Social NLP Experiments",
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")


if __name__ == "__main__":
    test()
