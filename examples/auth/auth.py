import os
import uuid

import mlflow.server


class User:
    MLFLOW_TRACKING_USERNAME = "MLFLOW_TRACKING_USERNAME"
    MLFLOW_TRACKING_PASSWORD = "MLFLOW_TRACKING_PASSWORD"

    def __init__(self, username, password) -> None:
        self.username = username
        self.password = password
        self.env = {}

    def _record_env_var(self, key):
        if key := os.getenv(key):
            self.env[key] = key

    def _restore_env_var(self, key):
        if value := self.env.get(key):
            os.environ[key] = value
        else:
            del os.environ[key]

    def __enter__(self):
        self._record_env_var(User.MLFLOW_TRACKING_USERNAME)
        self._record_env_var(User.MLFLOW_TRACKING_PASSWORD)
        os.environ[User.MLFLOW_TRACKING_USERNAME] = self.username
        os.environ[User.MLFLOW_TRACKING_PASSWORD] = self.password
        return self

    def __exit__(self, *_exc):
        self._restore_env_var(User.MLFLOW_TRACKING_USERNAME)
        self._restore_env_var(User.MLFLOW_TRACKING_PASSWORD)
        self.env.clear()


tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.server.get_app_client("basic-auth", tracking_uri)
A = User("user_a", "password_a")
B = User("user_b", "password_b")

with A:
    exp_a = mlflow.set_experiment(uuid.uuid4().hex)
    with mlflow.start_run():
        mlflow.log_metric("a", 1)

with B:
    mlflow.set_experiment(exp_a.name)
    try:
        with mlflow.start_run():  # not allowed
            mlflow.log_metric("b", 2)
    except Exception as e:
        print(str(e))

# Grant B permission to edit A's experiment
with A:
    client.create_experiment_permission(str(exp_a.experiment_id), B.username, "EDIT")

# B can edit now, should be able to log a metric
with B:
    mlflow.set_experiment(exp_a.name)
    with mlflow.start_run():
        mlflow.log_metric("b", 2)
