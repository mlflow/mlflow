"""
Example script that calls tracking APIs within / outside of a start_run() block. Also uses a
dependency not part of core MLflow (psutil), verifying that we run within a conda environment.
"""
import mlflow
import sys
import psutil


def call_tracking_apis():
    mlflow.log_metric("some_key", 3)
    mlflow.log_metric("cpu_percent", psutil.cpu_percent())


def main(use_start_run):
    if use_start_run:
        print("Running with start_run API")
        with mlflow.start_run():
            call_tracking_apis()
    else:
        print("Running without start_run API")
        call_tracking_apis()


if __name__ == "__main__":
    main(use_start_run=int(sys.argv[1]))
