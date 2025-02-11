# Location: mlflow/tracking/fluent.py:1327
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1327 '])
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
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))


if __name__ == "__main__":
    test()
