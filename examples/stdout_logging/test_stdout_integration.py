import time

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5002")

if __name__ == "__main__":
    mlflow.set_experiment("stdout_test")

    print("Testing stdout logging integration...")

    with mlflow.start_run(log_stdout=True, log_stdout_interval=3) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print("This should appear in both terminal and MLflow!")

        N_LOGS = 30
        for i in range(N_LOGS):
            print(f"Message {i + 1}/{N_LOGS}")
            time.sleep(1)

        print("Test completed!")

    print("This message should only appear in terminal (run has ended)")
