# Location: mlflow/tracking/fluent.py:512
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:512 '])
def test(_):
    import mlflow

    with mlflow.start_run() as run:
        mlflow.log_param("p", 0)

    run_id = run.info.run_id
    print(
        "run_id: {}; lifecycle_stage: {}".format(
            run_id, mlflow.get_run(run_id).info.lifecycle_stage
        )
    )


if __name__ == "__main__":
    test()
