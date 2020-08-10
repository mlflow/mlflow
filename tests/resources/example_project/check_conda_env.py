# Import a dependency in MLflow's setup.py that's in our conda.yaml but not included with MLflow
# by default, verify that we can use it.

import os
import sys

import psutil

import mlflow


def main(expected_env_name):
    actual_conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    assert actual_conda_env == expected_env_name, (
        "Script expected to be run from conda env %s but was actually run from env"
        " %s" % (expected_env_name, actual_conda_env)
    )
    mlflow.log_metric("CPU usage", psutil.cpu_percent())


if __name__ == "__main__":
    main(sys.argv[1])
