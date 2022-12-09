import mlflow
import random

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow server --backend-store-uri=sqlite:///mlflow.db
# MLFLOW_TRACKING_URI=sqlite:///mlflow.db ./dev/run-dev-server.sh
with mlflow.start_run() as run:
    for i in range(100):
        mlflow.log_metrics({"x": random.random(), "y": random.random()}, step=i)
