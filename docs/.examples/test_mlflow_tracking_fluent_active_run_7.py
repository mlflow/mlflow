# Location: mlflow/tracking/fluent.py:420
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:420 '])
def test(_):
    import mlflow

    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    mlflow.end_run()


if __name__ == "__main__":
    test()
