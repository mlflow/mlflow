# Location: mlflow/tracking/fluent.py:1170
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1170 '])
def test(_):
    import mlflow

    # Case sensitive name
    experiment = mlflow.get_experiment_by_name("Default")
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))


if __name__ == "__main__":
    test()
