# Location: mlflow/tracking/fluent.py:1138
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1138 '])
def test(_):
    import mlflow

    experiment = mlflow.get_experiment("0")
    print("Name: {}".format(experiment.name))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))


if __name__ == "__main__":
    test()
