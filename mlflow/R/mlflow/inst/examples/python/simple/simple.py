import os
import mlflow

if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.log_param("parameter", 5)
        mlflow.log_metric("metric", 0)
