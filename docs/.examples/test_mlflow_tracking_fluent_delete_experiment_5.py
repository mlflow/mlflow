# Location: mlflow/tracking/fluent.py:1366
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:1366 '])
def test(_):
    import mlflow

    experiment_id = mlflow.create_experiment("New Experiment")
    mlflow.delete_experiment(experiment_id)

    # Examine the deleted experiment details.
    experiment = mlflow.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Last Updated timestamp: {}".format(experiment.last_update_time))


if __name__ == "__main__":
    test()
