# Location: mlflow/tracking/fluent.py:797
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:797 '])
def test(_):
    import mlflow

    tags = {
        "engineering": "ML Platform",
        "release.candidate": "RC1",
        "release.version": "2.2.0",
    }

    # Set a batch of tags
    with mlflow.start_run():
        mlflow.set_experiment_tags(tags)


if __name__ == "__main__":
    test()
