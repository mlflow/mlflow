import mlflow
import os
import sys

path = sys.argv[1]

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "stream-s3"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    mlflow.log_artifact(path, "test")

client = mlflow.tracking.MlflowClient()
dst_path = client.download_artifacts(run.info.run_id, "test")

with open(os.path.join(dst_path, path)) as f:
    print(f.read())
