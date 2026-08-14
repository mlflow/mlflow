import os
import tempfile
from pprint import pprint

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient


def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def log_artifacts():
    # Upload artifacts
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path_a = os.path.join(tmp_dir, "a.txt")
        save_text(tmp_path_a, "0")
        tmp_sub_dir = os.path.join(tmp_dir, "dir")
        os.makedirs(tmp_sub_dir)
        tmp_path_b = os.path.join(tmp_sub_dir, "b.txt")
        save_text(tmp_path_b, "1")
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifacts(tmp_sub_dir, artifact_path="dir")
        return run.info.run_id


def main():
    assert "MLFLOW_TRACKING_URI" in os.environ

    # Log artifacts
    run_id1 = log_artifacts()
    # Download artifacts
    client = MlflowClient()
    print("Downloading artifacts")
    pprint(os.listdir(download_artifacts(run_id=run_id1, artifact_path="")))
    pprint(os.listdir(download_artifacts(run_id=run_id1, artifact_path="dir")))

    # List artifacts
    print("Listing artifacts")
    pprint(client.list_artifacts(run_id1))
    pprint(client.list_artifacts(run_id1, "dir"))

    # Log artifacts again
    run_id2 = log_artifacts()
    # Delete the run to test `mlflow gc` command
    client.delete_run(run_id2)
    print(f"Deleted run: {run_id2}")


if __name__ == "__main__":
    main()
