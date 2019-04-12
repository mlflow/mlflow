""" Example script that calls tracking APIs within / outside of a start_run() block. """
import mlflow
import os
import sys


def call_tracking_apis():
    mlflow.log_metric("some_key", 3)


def patch_run_yamls():  # Temporary patch while tranistioning to string experiment_id
    run_meta_file = os.path.join("/mlflow/tmp/mlruns/0/", os.environ["MLFLOW_RUN_ID"], "meta.yaml")
    with open(run_meta_file, "r") as stream:
        lines = stream.readlines()
    with open(run_meta_file, "w") as stream:
        for line in lines:
            if line.startswith("experiment_id"):
                stream.write(line.replace("'", ""))
            else:
                stream.write(line)


def main(use_start_run):
    print(mlflow.__version__)
    if mlflow.__version__ == "0.8.1":
        patch_run_yamls()  # TODO remove this after 1.0 release

    if use_start_run:
        print("Running with start_run API")
        with mlflow.start_run():
            call_tracking_apis()
    else:
        print("Running without start_run API")
        call_tracking_apis()


if __name__ == "__main__":
    main(use_start_run=int(sys.argv[1]))
