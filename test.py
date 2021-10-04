"""
$ mlflow server
$ export BUCKET="s3://my-bucket"
"""

import mlflow
import os
from glob import glob

os.environ["MLFLOW_ENABLE_PRESIGNED_URL"] = "true"

bucket_name = os.environ["BUCKET"]
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "presigned-url-test"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name, bucket_name)
mlflow.set_experiment(experiment_name)

# Upload
with mlflow.start_run() as run:
    mlflow.log_artifact("test.py", artifact_path="foo")

# Download
client = mlflow.tracking.MlflowClient()
local_path = client.download_artifacts(run.info.run_id, "foo", None)

with open(glob(local_path + "/*")[0]) as f:
    print(f.read())
