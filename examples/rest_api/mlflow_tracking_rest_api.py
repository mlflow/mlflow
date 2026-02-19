"""
This simple example shows how you could use MLflow REST API to create new
runs inside an experiment to log parameters/metrics.  Using MLflow REST API
instead of MLflow library might be useful to embed in an application where
you don't want to depend on the whole MLflow library, or to make
your own HTTP requests in another programming language (not Python).
For more details on MLflow REST API endpoints check the following page:

https://www.mlflow.org/docs/latest/rest-api.html
"""

import argparse
import os
import pwd

import requests

from mlflow.utils.time import get_current_time_millis

_DEFAULT_USER_ID = "unknown"


class MlflowTrackingRestApi:
    def __init__(self, hostname, port, experiment_id):
        self.base_url = "http://" + hostname + ":" + str(port) + "/api/2.0/mlflow"
        self.experiment_id = experiment_id
        self.run_id = self.create_run()

    def create_run(self):
        """Create a new run for tracking."""
        url = self.base_url + "/runs/create"
        # user_id is deprecated and will be removed from the API in a future release
        payload = {
            "experiment_id": self.experiment_id,
            "start_time": get_current_time_millis(),
            "user_id": _get_user_id(),
        }
        r = requests.post(url, json=payload)
        run_id = None
        if r.status_code == 200:
            run_id = r.json()["run"]["info"]["run_uuid"]
        else:
            print("Creating run failed!")
        return run_id

    def search_experiments(self):
        """Get all experiments."""
        url = self.base_url + "/experiments/search"
        r = requests.get(url)
        experiments = None
        if r.status_code == 200:
            experiments = r.json()["experiments"]
        return experiments

    def log_param(self, param):
        """Log a parameter dict for the given run."""
        url = self.base_url + "/runs/log-parameter"
        payload = {"run_id": self.run_id, "key": param["key"], "value": param["value"]}
        r = requests.post(url, json=payload)
        return r.status_code

    def log_metric(self, metric):
        """Log a metric dict for the given run."""
        url = self.base_url + "/runs/log-metric"
        payload = {
            "run_id": self.run_id,
            "key": metric["key"],
            "value": metric["value"],
            "timestamp": metric["timestamp"],
            "step": metric["step"],
        }
        r = requests.post(url, json=payload)
        return r.status_code


def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="MLflow REST API Example")

    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        dest="hostname",
        help="MLflow server hostname/ip (default: localhost)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        dest="port",
        help="MLflow server port number (default: 5000)",
    )

    parser.add_argument(
        "--experiment-id",
        type=int,
        default=0,
        dest="experiment_id",
        help="Experiment ID (default: 0)",
    )

    print("Running mlflow_tracking_rest_api.py")

    args = parser.parse_args()

    mlflow_rest = MlflowTrackingRestApi(args.hostname, args.port, args.experiment_id)
    # Parameter is a key/val pair (str types)
    param = {"key": "alpha", "value": "0.1980"}
    status_code = mlflow_rest.log_param(param)
    if status_code == 200:
        print(
            "Successfully logged parameter: {} with value: {}".format(param["key"], param["value"])
        )
    else:
        print("Logging parameter failed!")
    # Metric is a key/val pair (key/val have str/float types)
    metric = {
        "key": "precision",
        "value": 0.769,
        "timestamp": get_current_time_millis(),
        "step": 1,
    }
    status_code = mlflow_rest.log_metric(metric)
    if status_code == 200:
        print(
            "Successfully logged parameter: {} with value: {}".format(
                metric["key"], metric["value"]
            )
        )
    else:
        print("Logging metric failed!")
