# Location: mlflow/mlflow/tracking/fluent.py:1503
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1503 '])
def test(_):
    import mlflow

    with mlflow.start_run() as run:
        mlflow.log_param("p", 0)

    run_id = run.info.run_id
    mlflow.delete_run(run_id)

    print(
        f"run_id: {run_id}; lifecycle_stage: {mlflow.get_run(run_id).info.lifecycle_stage}"
    )


if __name__ == "__main__":
    test()
