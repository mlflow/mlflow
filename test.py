import mlflow
import uuid
import os
import requests
import base64


class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password

        self.u = None
        self.p = None

    @property
    def name(self):
        """
        Alias for username
        """
        return self.username

    def __enter__(self):
        self.u = os.environ.get("MLFLOW_TRACKING_USERNAME")
        self.p = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password

    def __exit__(self, *args):
        if self.u is None:
            del os.environ["MLFLOW_TRACKING_USERNAME"]
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.u
        if self.p is None:
            del os.environ["MLFLOW_TRACKING_PASSWORD"]
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.p


a = User("user_a@test.com", "password_a")
b = User("user_b@test.com", "password_b")


def should_fail(f):
    try:
        f()
    except Exception as e:
        print(e)
    else:
        raise Exception("Should not reach here")


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


name_a = f"a_{uuid.uuid4().hex}"
should_fail(lambda: mlflow.create_experiment(name_a))

with a:
    # This doesn't work
    exp_id_a = mlflow.create_experiment(name_a)

with b:
    name_b = f"b_{uuid.uuid4().hex}"
    exp_id_b = mlflow.create_experiment(name_b)

    should_fail(lambda: mlflow.get_experiment(exp_id_a))

    api(
        f"experiments/{exp_id_b}/permissions",
        "PUT",
        json={"user": a.name, "permission": "READ"},
    )

with a:
    mlflow.search_experiments()


# Allow B to read A's experiment
with a:
    api(
        f"experiments/{exp_id_a}/permissions",
        "PUT",
        json={"user": b.name, "permission": "READ"},
    )

    print(
        api(
            f"experiments/{exp_id_a}/permissions",
            "GET",
        )
    )


with b:
    # B can read
    mlflow.get_experiment(exp_id_a)

    # but not edit
    should_fail(lambda: mlflow.MlflowClient().rename_experiment(exp_id_a, "new_name"))


with a:
    api(
        f"experiments/{exp_id_a}/permissions",
        "POST",
        json={"user": b.name, "permission": "EDIT"},
    )

with b:
    new_name = f"a_{uuid.uuid4().hex}"
    # B can edit
    mlflow.MlflowClient().rename_experiment(exp_id_a, new_name)

    # but not delete
    should_fail(lambda: mlflow.MlflowClient().delete_experiment(exp_id_a))

    # B can't update permissions on A's experiments
    should_fail(
        lambda: api(
            f"experiments/{exp_id_a}/permissions",
            "POST",
            json={"user": b.name, "permission": "MANAGE"},
        )
    )


with a:
    api(
        f"experiments/{exp_id_a}/permissions",
        "DELETE",
        json={"user": b.name},
    )

with b:
    should_fail(lambda: mlflow.get_experiment(exp_id_a))

print("SUCCESS")
