# Location: mlflow/tracking/client.py:178
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/client.py:178 '])
def test(_):
    import mlflow
    from mlflow import MlflowClient

    # Create nested runs
    with mlflow.start_run():
        with mlflow.start_run(nested=True) as child_run:
            child_run_id = child_run.info.run_id

    client = MlflowClient()
    parent_run = client.get_parent_run(child_run_id)

    print("child_run_id: {}".format(child_run_id))
    print("parent_run_id: {}".format(parent_run.info.run_id))


if __name__ == "__main__":
    test()
