from pprint import pprint

from sklearn.linear_model import LogisticRegression
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run() as run:
    mlflow.log_param("p", 0)
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(LogisticRegression(), artifact_path="model")
    print("Artifact URI:", run.info.artifact_uri)

client = mlflow.tracking.MlflowClient()
pprint(client.list_artifacts(run.info.run_id))
pprint(client.list_artifacts(run.info.run_id, path="model"))
