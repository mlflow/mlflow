# Location: mlflow/tracking/fluent.py:94
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:94 '])
def test(_):
    import mlflow

    # Set an experiment name, which must be unique and case-sensitive.
    experiment = mlflow.set_experiment("Social NLP Experiments")

    # Get Experiment Details
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


if __name__ == "__main__":
    test()
