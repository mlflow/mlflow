import time

import mlflow

if __name__ == "__main__":
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run() as run:
        time.sleep(11)

    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
    print(mlflow_run.data.metrics)
