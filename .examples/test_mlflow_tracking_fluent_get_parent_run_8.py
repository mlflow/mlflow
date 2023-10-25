# Location: mlflow/mlflow/tracking/fluent.py:540
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:540 '])
def test(_):
    import mlflow

    # Create nested runs
    with mlflow.start_run():
        with mlflow.start_run(nested=True) as child_run:
            child_run_id = child_run.info.run_id

    parent_run = mlflow.get_parent_run(child_run_id)

    print(f"child_run_id: {child_run_id}")
    print(f"parent_run_id: {parent_run.info.run_id}")


if __name__ == "__main__":
    test()
