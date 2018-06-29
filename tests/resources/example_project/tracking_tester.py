import mlflow
import sys

if sys.argv[1] == "True"

with mlflow.start_run():
    mlflow.log_metric("metric_name", 3)
    mlflow.log_metric("metric_name", 4)
