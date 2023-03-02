import mlflow
import uuid
import os
import requests
import base64

USERNAME_A = "user_a@test.com"
PASSWORD_A = "password_a"
USERNAME_B = "user_b@test.com"
PASSWORD_B = "password_b"


def use_a():
    os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME_A
    os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD_A


def use_b():
    os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME_B
    os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD_B


TRACKING_URI = "http://localhost:5000"


def api(endpoint, method="GET", **kwargs):
    url = f"{TRACKING_URI}/api/2.0/mlflow/{endpoint}"
    u = os.environ["MLFLOW_TRACKING_USERNAME"]
    p = os.environ["MLFLOW_TRACKING_PASSWORD"]
    basic_auth_str = (f"{u}:{p}").encode("utf-8")
    auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    headers = {"Authorization": auth_str, "Content-Type": "application/json"}
    response = requests.request(method, url, headers=headers, **kwargs)
    response.raise_for_status()
    return response.text


mlflow.set_tracking_uri(TRACKING_URI)


use_a()
name_a = uuid.uuid4().hex
# This doesn't work
exp_id_a = mlflow.create_experiment(name_a)

use_b()
name_b = uuid.uuid4().hex
exp_id_b = mlflow.create_experiment(name_b)

try:
    mlflow.get_experiment(exp_id_a)
except Exception as e:
    print(e)
else:
    raise Exception("Should not reach here")


# Allow B to read A's experiment
use_a()
api(
    f"experiments/{exp_id_a}/permissions",
    "PUT",
    json={"user": USERNAME_B, "access_level": "CAN_READ"},
)

print(mlflow.get_experiment(exp_id_a))

api(
    f"experiments/{exp_id_a}/permissions",
    "POST",
    json={"user": USERNAME_B, "access_level": "CAN_"},
)
