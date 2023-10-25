# Location: mlflow/mlflow/tracking/fluent.py:1516
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1516 '])
def test(_):
    import mlflow

    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", "w") as f:
        f.write(features)

    # Log the artifact in a directory "features" under the root artifact_uri/features
    with mlflow.start_run():
        mlflow.log_artifact("features.txt", artifact_path="features")

        # Fetch the artifact uri root directory
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")

        # Fetch a specific artifact uri
        artifact_uri = mlflow.get_artifact_uri(artifact_path="features/features.txt")
        print(f"Artifact uri: {artifact_uri}")


if __name__ == "__main__":
    test()
