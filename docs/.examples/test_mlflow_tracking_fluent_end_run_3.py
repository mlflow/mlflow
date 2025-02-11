# Location: mlflow/tracking/fluent.py:374
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:374 '])
def test(_):
    import mlflow

    # Start run and get status
    mlflow.start_run()
    run = mlflow.active_run()
    print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))

    # End run and get status
    mlflow.end_run()
    run = mlflow.get_run(run.info.run_id)
    print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
    print("--")

    # Check for any active runs
    print("Active run: {}".format(mlflow.active_run()))


if __name__ == "__main__":
    test()
