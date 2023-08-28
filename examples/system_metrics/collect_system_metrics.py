import time

import mlflow

if __name__ == "__main__":
    run = mlflow.start_run()
    time.sleep(11)
    mlflow.end_run()
    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
    print(mlflow_run.data.metrics)
