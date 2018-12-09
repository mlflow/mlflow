"""
This simple example shows how we should use MLFlow Rest API for creating new runs and log parameters/metrics.
For more details see: https://www.mlflow.org/docs/latest/rest-api.html 
"""


import os
import time
import requests

#change this base URL so that it points to the host where your MLFlow server is running
BASE_URL = 'http://localhost:5000/api/2.0/preview/mlflow'

_DEFAULT_USER_ID = "unknown"


def create_run(experiment_id):
	"""Create a new run for tracking."""
	url = BASE_URL + '/runs/create'
	payload = {'experiment_id': experiment_id, 'start_time': int(time.time() * 1000), 'user_id': _get_user_id()}
	r = requests.post(url, json=payload)
	run_id = None
	if r.status_code == 200:
		run_id = r.json()['run']['info']['run_uuid']
	return run_id


def list_experiments():
	"""Get all experiments."""
	url = BASE_URL + '/experiments/list'
	r = requests.get(url)
	experiments = None
	if r.status_code == 200:
		experiments = r.json()['experiments']
	return experiments


def log_param(run_id, param):
	"""Log a parameter dict for the given run."""
	url = BASE_URL + '/runs/log-parameter'
	payload = {'run_uuid': run_id, 'key': param['key'], 'value': param['value']}
	r = requests.post(url, json=payload)
	return r.status_code


def log_metric(run_id, metric):
	"""Log a metric dict for the given run."""
	url = BASE_URL + '/runs/log-metric'
	payload = {'run_uuid': run_id, 'key': metric['key'], 'value': metric['value']}
	r = requests.post(url, json=payload)
	return r.status_code


def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID


if __name__ == "__main__":
	print("Running mlflow_tracking.py")
	run_id = create_run(0)
	# parameter is a key/val pair (str types)
	param = {'key': 'alpha', 'value': '0.5'}
	log_param(run_id, param)
	# metric is a key/val pair (key/val have str/float types)
	metric = {'key': 'precision', 'value': 0.769}	
	log_metric(run_id, metric)
