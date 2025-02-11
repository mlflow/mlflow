# Location: mlflow/tracking/fluent.py:652
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:652 '])
def test(_):
    import mlflow

    tags = {"engineering": "ML Platform", "engineering_remote": "ML Platform"}

    with mlflow.start_run() as run:
        mlflow.set_tags(tags)

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.delete_tag("engineering_remote")


if __name__ == "__main__":
    test()
