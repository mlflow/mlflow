from mlflow.system_metrics import SystemMetricsMonitor
import mlflow
import time

if __name__ == "__main__":
    with mlflow.start_run() as run:
        system_monitor = SystemMetricsMonitor(run, 1)

    system_monitor.start()
    time.sleep(10)
    system_monitor.finish()

    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
