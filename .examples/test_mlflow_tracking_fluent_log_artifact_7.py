# Location: mlflow/mlflow/tracking/fluent.py:857
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:857 '])
def test(_):
    import mlflow

    # Create a features.txt artifact file
    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", "w") as f:
        f.write(features)

    # With artifact_path=None write features.txt under
    # root artifact_uri/artifacts directory
    with mlflow.start_run():
        mlflow.log_artifact("features.txt")


if __name__ == "__main__":
    test()
