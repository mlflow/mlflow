import os
import tempfile
from pprint import pprint

from sklearn.linear_model import LogisticRegression

import mlflow


def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    mlflow.set_tracking_uri("http://localhost:5000")

    # Upload
    with mlflow.start_run() as run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "test.txt")
            save_text(tmp_path, "test")
            mlflow.log_artifact(tmp_path)
            mlflow.log_artifacts(tmp_dir, artifact_path="subdir")

        mlflow.sklearn.log_model(LogisticRegression(), artifact_path="model")

    # Download
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run.info.run_id, "subdir")
    pprint(os.listdir(local_path))

    # List
    pprint(client.list_artifacts(run.info.run_id, path="model"))


if __name__ == "__main__":
    main()

