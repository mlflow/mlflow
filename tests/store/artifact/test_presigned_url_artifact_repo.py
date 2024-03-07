import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository


def test_download_artifacts():
    mlflow.set_registry_uri("databricks-uc://arclight-prototype")

    # uri = mlflow.MlflowClient().get_model_version_download_uri("artjen.test.newtestmodel", "2")
    # ModelsArtifactRepository(uri).download_artifacts(artifact_path="")
    ModelsArtifactRepository("models:/artjen.test.newtestmodel/2").download_artifacts(artifact_path="")


def test_upload_artifacts(tmp_path):
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier

    mlflow.set_registry_uri("databricks-uc://arclight-prototype")

    with mlflow.start_run():
        # Train a sklearn model on the iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)
        clf = RandomForestClassifier(max_depth=7)
        clf.fit(X, y)
        # Take the first row of the training dataset as the model input example.
        input_example = X.iloc[[0]]
        # Log the model and register it as a new version in UC.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            # The signature is automatically inferred from the input example and its predicted output.
            input_example=input_example,
            registered_model_name="artjen.test.newtestmodel",
        )
