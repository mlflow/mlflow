# Location: mlflow/mlflow/tracking/client.py:141
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/client.py:141 '])
def test(_):
    import mlflow
    from mlflow import MlflowClient

    # Create nested runs
    with mlflow.start_run():
        with mlflow.start_run(nested=True) as child_run:
            child_run_id = child_run.info.run_id

    client = MlflowClient()
    parent_run = client.get_parent_run(child_run_id)

    print(f"child_run_id: {child_run_id}")
    print(f"parent_run_id: {parent_run.info.run_id}")


if __name__ == "__main__":
    test()
